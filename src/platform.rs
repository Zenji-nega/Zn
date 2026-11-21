//! Things only the operating system can do for us.
//!
//! On Android, reading a model file from shared storage needs the "All files access"
//! permission, which is granted on a system settings page. We open that page and
//! later check whether the user allowed it. Both go through JNI, in Rust.
//! On other systems there is nothing to ask for.

use std::path::PathBuf;

/// Where file browsing starts.
pub fn browse_start() -> PathBuf {
    for candidate in ["/storage/emulated/0", "/sdcard"] {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return path;
        }
    }
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// Handy starting points shown above the file list.
pub fn shortcuts() -> Vec<(&'static str, PathBuf)> {
    let mut places = Vec::new();
    let root = PathBuf::from("/storage/emulated/0");
    if root.is_dir() {
        places.push(("Internal storage", root.clone()));
        for (label, sub) in [("Downloads", "Download"), ("Documents", "Documents")] {
            if root.join(sub).is_dir() {
                places.push((label, root.join(sub)));
            }
        }
    } else if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let downloads = home.join("Downloads");
        places.push(("Home", home));
        if downloads.is_dir() {
            places.push(("Downloads", downloads));
        }
    }
    places
}

#[cfg(target_os = "android")]
pub use android::{has_storage_access, init_android, request_storage_access};

#[cfg(not(target_os = "android"))]
pub fn has_storage_access() -> bool {
    true
}

#[cfg(not(target_os = "android"))]
pub fn request_storage_access() -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "android")]
mod android {
    use std::sync::OnceLock;

    use jni::{
        JNIEnv, JavaVM,
        objects::{JObject, JString, JValue},
    };

    /// (JavaVM pointer, Activity pointer), as integers so they can live in a static.
    static HANDLES: OnceLock<(usize, usize)> = OnceLock::new();

    /// Remembers the Java VM and activity. Call once from `android_main`.
    pub fn init_android(app: &android_activity::AndroidApp) {
        let _ = HANDLES.set((app.vm_as_ptr() as usize, app.activity_as_ptr() as usize));
    }

    fn with_env<R>(
        work: impl FnOnce(&mut JNIEnv, &JObject) -> jni::errors::Result<R>,
    ) -> Result<R, String> {
        let &(vm, activity) = HANDLES.get().ok_or("the app is not ready yet")?;
        // SAFETY: both pointers come from android-activity and stay valid for the process lifetime.
        let vm =
            unsafe { JavaVM::from_raw(vm as *mut jni::sys::JavaVM) }.map_err(|e| e.to_string())?;
        let mut env = vm.attach_current_thread().map_err(|e| e.to_string())?;
        // SAFETY: as above; the activity global reference outlives this call.
        let activity = unsafe { JObject::from_raw(activity as jni::sys::jobject) };
        let result = work(&mut env, &activity);
        if env.exception_check().unwrap_or(false) {
            let _ = env.exception_clear();
            return Err("the system refused the request".into());
        }
        result.map_err(|e| e.to_string())
    }

    fn sdk_int(env: &mut JNIEnv) -> jni::errors::Result<i32> {
        env.get_static_field("android/os/Build$VERSION", "SDK_INT", "I")?
            .i()
    }

    /// True when the app may read files anywhere in shared storage.
    pub fn has_storage_access() -> bool {
        with_env(|env, _| {
            // Before Android 11 there is no such switch; ordinary storage permissions apply.
            if sdk_int(env)? < 30 {
                return Ok(true);
            }
            env.call_static_method(
                "android/os/Environment",
                "isExternalStorageManager",
                "()Z",
                &[],
            )?
            .z()
        })
        .unwrap_or(false)
    }

    /// Opens the system page where the user can allow "All files access" for this app.
    pub fn request_storage_access() -> Result<(), String> {
        with_env(|env, activity| {
            let package: JString = env
                .call_method(activity, "getPackageName", "()Ljava/lang/String;", &[])?
                .l()?
                .into();
            let package: String = env.get_string(&package)?.into();
            let uri_text = env.new_string(format!("package:{package}"))?;
            let uri = env
                .call_static_method(
                    "android/net/Uri",
                    "parse",
                    "(Ljava/lang/String;)Landroid/net/Uri;",
                    &[JValue::Object(&uri_text)],
                )?
                .l()?;
            let action =
                env.new_string("android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION")?;
            let intent = env.new_object(
                "android/content/Intent",
                "(Ljava/lang/String;Landroid/net/Uri;)V",
                &[JValue::Object(&action), JValue::Object(&uri)],
            )?;
            env.call_method(
                activity,
                "startActivity",
                "(Landroid/content/Intent;)V",
                &[JValue::Object(&intent)],
            )?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browsing_has_a_starting_point_and_desktop_needs_no_permission() {
        assert!(!browse_start().as_os_str().is_empty());
        assert!(has_storage_access());
        assert!(request_storage_access().is_ok());
    }
}
