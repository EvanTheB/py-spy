//! Support for attributing samples to trio tasks.
//!
//! A trio program runs every task on a single thread, so a plain stack trace only
//! shows the task that happens to be stepping when the sample was taken, under the
//! generic `trio.run -> unrolled_run` frames. This module:
//!
//! * finds the trio `Runner` from the `run`/`unrolled_run` frames on the thread stack,
//! * replaces the `unrolled_run` frame of the running task with synthetic frames naming
//!   the task and its ancestors (following `Task._parent_nursery._parent_task`),
//! * optionally builds a stack for every *waiting* task by walking its suspended
//!   coroutine `await` chain, so wall clock time spent waiting can be profiled too.
//!
//! Reading coroutine objects depends on CPython internals, so this is only supported on
//! Python 3.15.

use std::collections::HashMap;
use std::mem::size_of;

use anyhow::{Context, Error, Result};
use remoteprocess::ProcessMemory;

use crate::config::LineNo;
use crate::python_bindings::v3_15_0;
use crate::python_data_access::{copy_string, DictIterator, PY_TPFLAGS_MANAGED_DICT};
use crate::python_interpreters::{CodeObject, FrameObject};
use crate::stack_trace::{get_line_number, get_locals, Frame, StackTrace};
use crate::version::Version;

const TRIO_RUN_FILE: &str = "trio/_core/_run.py";

// PyFrameState values (Include/internal/pycore_frame.h)
const FRAME_SUSPENDED_YIELD_FROM: i8 = 2;
const FRAME_SUSPENDED_YIELD_FROM_LOCKED: i8 = 3;
const FRAME_EXECUTING: i8 = 4;
const FRAME_CLEARED: i8 = 5;

// Limits guarding against torn reads producing garbage
const MAX_TASKS: usize = 100_000;
const MAX_AWAIT_DEPTH: usize = 1024;
const MAX_NURSERY_DEPTH: usize = 1024;

/// Initial segment shared by generator, coroutine and async generator objects
/// (_PyGenObject_HEAD in Include/internal/pycore_interpframe_structs.h)
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
struct PyGenObjectHead {
    ob_base: v3_15_0::PyObject,
    weakreflist: usize,
    name: usize,
    qualname: usize,
    exc_state: v3_15_0::_PyErr_StackItem,
    origin_or_finalizer: usize,
    hooks_inited: u8,
    closed: u8,
    running_async: u8,
    frame_state: i8,
    iframe: v3_15_0::_PyInterpreterFrame,
}

/// Leading fields of PySetObject (Include/cpython/setobject.h)
#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
struct PySetObjectHead {
    ob_base: v3_15_0::PyObject,
    fill: isize,
    used: isize,
    mask: isize,
    table: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
struct SetEntry {
    key: usize,
    hash: isize,
}

/// The name and parent of a trio Task
#[derive(Clone)]
struct TaskInfo {
    name: String,
    parent: Option<usize>,
}

pub struct TrioInspector {
    version: Version,
    /// type address -> (slot name -> offset), for __slots__ classes like Task and Runner
    slot_offsets: HashMap<usize, HashMap<String, usize>>,
    /// type address -> tp_name
    type_names: HashMap<usize, String>,
}

impl TrioInspector {
    pub fn new(version: &Version) -> Result<TrioInspector, Error> {
        if version.major != 3 || version.minor != 15 {
            return Err(format_err!(
                "--trio is only supported on Python 3.15 (found {})",
                version
            ));
        }
        Ok(TrioInspector {
            version: version.clone(),
            slot_offsets: HashMap::new(),
            type_names: HashMap::new(),
        })
    }

    /// Rewrites `trace` so the running trio task (if any) is shown by name, and if
    /// `include_waiting` is set returns an extra trace for each waiting trio task.
    ///
    /// `frame_ptr` is the address of the innermost _PyInterpreterFrame of the thread
    /// that `trace` was collected from.
    pub fn annotate<P: ProcessMemory>(
        &mut self,
        process: &P,
        frame_ptr: usize,
        trace: &mut StackTrace,
        lineno: LineNo,
        include_waiting: bool,
    ) -> Result<Vec<StackTrace>, Error> {
        let (runner, current) = match find_runner(process, frame_ptr)? {
            Some(found) => found,
            None => return Ok(Vec::new()),
        };

        let mut tasks: HashMap<usize, TaskInfo> = HashMap::new();

        // The `task` local of unrolled_run is left pointing at the last task stepped, so
        // only treat it as running if its coroutine is actually executing.
        let mut running = None;
        if let Some(task) = current {
            let coro = self.get_attr(process, task, "coro")?;
            let gen: PyGenObjectHead = process.copy_struct(coro)?;
            if gen.frame_state == FRAME_EXECUTING {
                running = Some(task);
            }
        }

        if let Some(task) = running {
            let chain = self.task_chain(process, task, &mut tasks)?;
            trace.trio_task = Some(tasks[&task].name.clone());
            if let Some(i) = trace
                .frames
                .iter()
                .position(|f| is_trio_frame(f, "unrolled_run"))
            {
                trace.frames.splice(i..=i, chain);
            }
        }

        let mut waiting = Vec::new();
        if !include_waiting {
            return Ok(waiting);
        }

        // Frames from trio.run() outwards, shared by every waiting task
        let outer: &[Frame] = match trace.frames.iter().position(|f| is_trio_frame(f, "run")) {
            Some(i) => &trace.frames[i..],
            None => &[],
        };

        let task_set = self.get_attr(process, runner, "tasks")?;
        for task in self.set_items(process, task_set)? {
            if Some(task) == running {
                continue;
            }
            let coro = self.get_attr(process, task, "coro")?;
            let mut frames = self
                .coroutine_stack(process, coro, lineno)
                .unwrap_or_else(|e| {
                    warn!("Failed to get trio task stack: {}", e);
                    Vec::new()
                });
            let chain = self.task_chain(process, task, &mut tasks)?;
            if chain.is_empty() {
                // the <init> task: it isn't useful to show on its own
                continue;
            }
            frames.extend(chain);
            frames.extend_from_slice(outer);

            waiting.push(StackTrace {
                frames,
                active: false,
                owns_gil: false,
                trio_task: Some(tasks[&task].name.clone()),
                ..trace.clone()
            });
        }
        Ok(waiting)
    }

    /// Returns synthetic frames for a task and its ancestors, innermost first. The root
    /// `<init>` task that trio creates internally is omitted.
    fn task_chain<P: ProcessMemory>(
        &mut self,
        process: &P,
        task: usize,
        cache: &mut HashMap<usize, TaskInfo>,
    ) -> Result<Vec<Frame>, Error> {
        let mut frames = Vec::new();
        let mut current = Some(task);
        while let Some(task) = current {
            if frames.len() > MAX_NURSERY_DEPTH {
                return Err(format_err!("Max trio nursery depth reached"));
            }
            let info = match cache.get(&task) {
                Some(info) => info.clone(),
                None => {
                    let info = self.task_info(process, task)?;
                    cache.insert(task, info.clone());
                    info
                }
            };
            if info.parent.is_none() && info.name == "<init>" {
                break;
            }
            frames.push(task_frame(&info.name));
            current = info.parent;
        }
        Ok(frames)
    }

    fn task_info<P: ProcessMemory>(&mut self, process: &P, task: usize) -> Result<TaskInfo, Error> {
        let name_addr = self.get_attr(process, task, "name")?;
        let name = copy_string(name_addr as *const v3_15_0::PyUnicodeObject, process)
            .context("Failed to copy trio task name")?;
        let nursery = self.get_attr(process, task, "_parent_nursery")?;
        let parent = if nursery == 0 || self.type_name(process, nursery)? == "NoneType" {
            None
        } else {
            Some(self.get_attr(process, nursery, "_parent_task")?)
        };
        Ok(TaskInfo { name, parent })
    }

    /// Walks a suspended coroutine's await chain, returning frames innermost first
    fn coroutine_stack<P: ProcessMemory>(
        &mut self,
        process: &P,
        coro: usize,
        lineno: LineNo,
    ) -> Result<Vec<Frame>, Error> {
        let mut frames = Vec::new();
        let mut current = coro;
        loop {
            if frames.len() > MAX_AWAIT_DEPTH {
                return Err(format_err!("Max await depth reached"));
            }
            let gen: PyGenObjectHead = process
                .copy_struct(current)
                .context("Failed to copy coroutine")?;
            if gen.frame_state >= FRAME_CLEARED {
                break;
            }
            let frame = gen.iframe;
            let code = process
                .copy_pointer(frame.code())
                .context("Failed to copy PyCodeObject")?;
            let filename =
                copy_string(code.filename(), process).context("Failed to copy filename")?;
            let name = match code.qualname() {
                Some(qualname) => copy_string(qualname, process),
                None => copy_string(code.name(), process),
            }
            .context("Failed to copy function name")?;
            let line = match lineno {
                LineNo::NoLine => 0,
                LineNo::First => code.first_lineno(),
                LineNo::LastInstruction => {
                    get_line_number(&code, frame.lasti(), process).unwrap_or(0)
                }
            };
            frames.push(Frame {
                name,
                filename,
                module: None,
                short_filename: None,
                line,
                locals: None,
                is_entry: true,
                is_shim_entry: false,
            });

            if gen.frame_state != FRAME_SUSPENDED_YIELD_FROM
                && gen.frame_state != FRAME_SUSPENDED_YIELD_FROM_LOCKED
            {
                break;
            }
            // While suspended in `yield from`/`await`, python 3.15 keeps the object being
            // awaited one below the top of the frame's value stack (_PyFrame_StackPeek(f, 2))
            let top = frame.stackpointer as usize - 2 * size_of::<v3_15_0::_PyStackRef>();
            let awaited: usize = process
                .copy_struct(top)
                .context("Failed to copy awaited object")?;
            let awaited = awaited & !(v3_15_0::Py_TAG_BITS as usize);
            match self
                .type_name(process, awaited)
                .with_context(|| format!("Failed to get type of awaited object {:#x}", awaited))?
                .as_str()
            {
                "coroutine" | "generator" => current = awaited,
                _ => break,
            }
        }
        frames.reverse();
        Ok(frames)
    }

    /// Returns the addresses of all items in a python set
    fn set_items<P: ProcessMemory>(
        &mut self,
        process: &P,
        addr: usize,
    ) -> Result<Vec<usize>, Error> {
        let set: PySetObjectHead = process.copy_struct(addr)?;
        let size = set.mask as usize + 1;
        if set.mask < 0 || size > 4 * MAX_TASKS {
            return Err(format_err!("Invalid set size {}", size));
        }
        let entries: Vec<SetEntry> = process
            .copy(set.table, size * size_of::<SetEntry>())?
            .chunks_exact(size_of::<SetEntry>())
            .map(|b| unsafe { std::ptr::read_unaligned(b.as_ptr() as *const SetEntry) })
            .collect();

        let mut ret = Vec::new();
        for entry in entries {
            // skip empty slots and the dummy object marking deleted entries
            if entry.key == 0 || self.type_name(process, entry.key)? != "Task" {
                continue;
            }
            ret.push(entry.key);
        }
        Ok(ret)
    }

    /// Gets an attribute of an object, from either its __slots__ or its __dict__
    fn get_attr<P: ProcessMemory>(
        &mut self,
        process: &P,
        obj: usize,
        name: &str,
    ) -> Result<usize, Error> {
        let object: v3_15_0::PyObject = process.copy_struct(obj)?;
        let type_addr = object.ob_type as usize;

        if let Some(offset) = self.slot_offset(process, type_addr, name)? {
            return Ok(process.copy_struct(obj + offset)?);
        }

        let tp: v3_15_0::PyTypeObject = process.copy_struct(type_addr)?;
        let flags = tp.tp_flags as usize;
        let dict = if flags & PY_TPFLAGS_MANAGED_DICT != 0 {
            DictIterator::from_managed_dict(process, &self.version, obj, type_addr, flags)?
        } else if tp.tp_dictoffset > 0 {
            let dict_addr: usize = process.copy_struct(obj + tp.tp_dictoffset as usize)?;
            DictIterator::from(process, &self.version, dict_addr)?
        } else {
            return Err(format_err!("object has no attribute {}", name));
        };
        for entry in dict {
            let (key, value) = entry?;
            if copy_string(key as *const v3_15_0::PyUnicodeObject, process)? == name {
                return Ok(value);
            }
        }
        Err(format_err!("object has no attribute {}", name))
    }

    fn slot_offset<P: ProcessMemory>(
        &mut self,
        process: &P,
        type_addr: usize,
        name: &str,
    ) -> Result<Option<usize>, Error> {
        if !self.slot_offsets.contains_key(&type_addr) {
            let tp: v3_15_0::PyTypeObject = process.copy_struct(type_addr)?;
            let mut offsets = HashMap::new();
            let mut member = tp.tp_members as usize;
            while member != 0 && offsets.len() < 256 {
                let def: v3_15_0::PyMemberDef = process.copy_struct(member)?;
                if def.name.is_null() {
                    break;
                }
                offsets.insert(copy_cstr(process, def.name as usize)?, def.offset as usize);
                member += size_of::<v3_15_0::PyMemberDef>();
            }
            self.slot_offsets.insert(type_addr, offsets);
        }
        Ok(self.slot_offsets[&type_addr].get(name).copied())
    }

    fn type_name<P: ProcessMemory>(&mut self, process: &P, obj: usize) -> Result<String, Error> {
        let object: v3_15_0::PyObject = process.copy_struct(obj)?;
        let type_addr = object.ob_type as usize;
        if let Some(name) = self.type_names.get(&type_addr) {
            return Ok(name.clone());
        }
        let tp: v3_15_0::PyTypeObject = process.copy_struct(type_addr)?;
        let name = copy_cstr(process, tp.tp_name as usize)?;
        self.type_names.insert(type_addr, name.clone());
        Ok(name)
    }
}

/// Searches the thread's frames for trio's run loop, returning the address of the
/// `Runner` and the `task` local of `unrolled_run` (the task most recently stepped).
fn find_runner<P: ProcessMemory>(
    process: &P,
    mut frame_ptr: usize,
) -> Result<Option<(usize, Option<usize>)>, Error> {
    let mut current_task = None;
    let mut visited = 0;
    while frame_ptr != 0 && visited < 4096 {
        visited += 1;
        let frame_addr = frame_ptr as *const v3_15_0::_PyInterpreterFrame;
        let frame = process.copy_pointer(frame_addr)?;
        frame_ptr = frame.back() as usize;
        if !frame.is_python_frame() {
            continue;
        }
        let code = process.copy_pointer(frame.code())?;
        let name = copy_string(code.name(), process).unwrap_or_default();
        if name != "unrolled_run" && name != "run" {
            continue;
        }
        let filename = copy_string(code.filename(), process).unwrap_or_default();
        if !filename.ends_with(TRIO_RUN_FILE) {
            continue;
        }
        let locals = get_locals(&code, frame_addr, &frame, process)?;
        let local = |n: &str| locals.iter().find(|l| l.name == n).map(|l| l.addr);
        if name == "unrolled_run" {
            current_task = local("task");
        }
        if let Some(runner) = local("runner") {
            return Ok(Some((runner, current_task)));
        }
    }
    Ok(None)
}

fn is_trio_frame(frame: &Frame, name: &str) -> bool {
    frame.name == name && frame.filename.ends_with(TRIO_RUN_FILE)
}

fn task_frame(name: &str) -> Frame {
    Frame {
        name: format!("<trio task {}>", name),
        filename: String::new(),
        module: None,
        short_filename: None,
        line: 0,
        locals: None,
        is_entry: true,
        is_shim_entry: true,
    }
}

/// Copies a NUL terminated C string, reading at most 256 bytes
fn copy_cstr<P: ProcessMemory>(process: &P, addr: usize) -> Result<String, Error> {
    let mut bytes = Vec::new();
    let mut addr = addr;
    while bytes.len() < 256 {
        // read up to the end of the page, so we don't fault on a short string at its end
        let chunk = 64.min(4096 - (addr % 4096));
        let data = process.copy(addr, chunk)?;
        if let Some(end) = data.iter().position(|&b| b == 0) {
            bytes.extend_from_slice(&data[..end]);
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.extend_from_slice(&data);
        addr += chunk;
    }
    Err(format_err!("C string too long"))
}
