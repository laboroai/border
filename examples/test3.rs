use numpy::PyArray1;
use pyo3::prelude::{PyResult, Python};
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let np = py.import("numpy")?;
        let locals = [("np", np)].into_py_dict(py);
        let pyarray: &PyArray1<i32> = py
            .eval("np.absolute(np.array([-1, -2, -3], dtype='int32'))", Some(locals), None)?
            .extract()?;
        let readonly = pyarray.readonly();
        let slice = readonly.as_slice()?;
        assert_eq!(slice, &[1, 2, 3]);
        Ok(())
    })
}