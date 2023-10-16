//! Download pretrained model from a network drive.
// use core::slice::SlicePattern;
use anyhow::{Context, Result};
use log::info;
use reqwest::IntoUrl;
use std::{fmt::Debug, fs::File, io::Write, path::Path};

/// Download a pretrained model as a zip file from given url.
///
/// This function will download a zip file in `~/border/model`, extract it, then
/// return a path to the extracted directory, which should contains a pretrained
/// model.
pub fn get_model_from_url<T: AsRef<Path>>(
    url: impl IntoUrl + Debug,
    file_base: T,
) -> Result<impl AsRef<Path>> {
    info!("Download file from {:?}", url);
    let response = reqwest::blocking::get(url)?;

    let mut path_dir = dirs::home_dir().context("Couldn't find home directory")?;
    path_dir.push(".border/model/");
    let mut path_zip = path_dir.clone();
    path_zip.push(&file_base);
    path_zip.set_extension("zip");

    if path_zip.as_path().exists() {
        info!("Exists zip file {:?}, skips download", path_zip);
        path_dir.push(&file_base);
        return Ok(path_dir);
    }

    let mut zip_file = File::create(&path_zip.as_path()).context(format!(
        "Failed to create file {:?}\n\
             You may need to create directory: \"mkdir ~/.border/model\"
            ",
        path_zip
    ))?;

    // TODO: skip download when the zip file exists
    info!("Downloaded file as {:?}", path_zip.as_path());
    let content = response.bytes()?;
    zip_file.write_all(&content)?;
    zip_file.flush()?;

    info!("Unzip the file");
    let zip_file = File::open(&path_zip.as_path())?;
    let mut archive = zip::ZipArchive::new(zip_file)?;
    archive.extract(&path_dir.as_path())?;

    path_dir.push(&file_base);
    Ok(path_dir)
}

#[cfg(test)]
mod tests {
    use super::get_model_from_url;
    use anyhow::{Context, Result};
    use log::info;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_get_model_from_url() -> Result<()> {
        init();

        let url =
            "https://drive.google.com/uc?export=download&id=1TF5aN9fH5wd4APFHj9RP1JxuVNoi6lqJ";
        let file_base = "dqn_PongNoFrameskip-v4_20210428_ec2";

        let mut path = dirs::home_dir().context("Couldn't find home directory")?;
        path.push(".border/model");
        let model_root_dir = path.as_path();

        if !model_root_dir.exists() {
            info!("Create directory {:?}", model_root_dir);
            std::fs::create_dir_all(model_root_dir)?;
        }

        let mut path = dirs::home_dir().context("Couldn't find home directory")?;
        path.push(".border/model/");
        path.push(file_base);
        path.set_extension("zip");

        // ignore when failed to remove file
        std::fs::remove_file(&path.as_path()).unwrap_or(());

        let _path = get_model_from_url(url, file_base)?;

        Ok(())
    }
}
