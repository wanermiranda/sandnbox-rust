use cached_path::{Cache, Error, ProgressBar};
use std::path::PathBuf;

/// It downloads a file from a URL to a cache directory
///
/// Arguments:
///
/// * `src`: The URL of the file to download.
///
/// Returns:
///
/// A path to the cached file.
///
pub fn download_file_to_cache(src: &str) -> Result<PathBuf, Error> {
    let mut cache_dir = dirs::home_dir().unwrap();
    cache_dir.push(".cache");

    let cached_path = Cache::builder()
        .dir(cache_dir)
        .progress_bar(Some(ProgressBar::Light))
        .build()?
        .cached_path(src)?;
    Ok(cached_path)
}
