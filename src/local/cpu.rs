//! Picks the fast cores of a big.LITTLE phone for inference.
//!
//! Matrix multiplies are split evenly across worker threads, so one slow core makes
//! every step wait for it. Using only the fast cores (and keeping the workers at a
//! lower priority than the UI thread) is much quicker and keeps the screen responsive.

use std::sync::OnceLock;

/// Cores whose maximum frequency is within 80% of the fastest one.
/// `freqs` holds `(core index, max frequency)`.
pub fn big_cores(freqs: &[(usize, u64)]) -> Vec<usize> {
    let Some(max) = freqs.iter().map(|&(_, f)| f).max().filter(|&f| f > 0) else {
        return Vec::new();
    };
    freqs
        .iter()
        .filter(|&&(_, f)| f * 10 >= max * 8)
        .map(|&(core, _)| core)
        .collect()
}

fn read_freqs() -> Vec<(usize, u64)> {
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
    (0..cores)
        .filter_map(|core| {
            let path = format!("/sys/devices/system/cpu/cpu{core}/cpufreq/cpuinfo_max_freq");
            let freq = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;
            Some((core, freq))
        })
        .collect()
}

/// Fast cores, or empty when the phone does not tell us (or all cores are alike).
pub fn fast_cores() -> &'static [usize] {
    static CORES: OnceLock<Vec<usize>> = OnceLock::new();
    CORES.get_or_init(|| {
        let freqs = read_freqs();
        let big = big_cores(&freqs);
        if big.is_empty() || big.len() == freqs.len() {
            Vec::new()
        } else {
            big
        }
    })
}

/// How many worker threads to use when the user left it on automatic.
pub fn default_threads(total_cores: usize) -> usize {
    match fast_cores().len() {
        0 => (total_cores / 2).max(2).min(total_cores.max(1)),
        n => n.clamp(2, 6),
    }
}

/// Restricts the calling thread to the fast cores and lowers its priority a little.
/// Failures are ignored: this is only an optimization.
pub fn prefer_fast_cores() {
    #[cfg(any(target_os = "android", target_os = "linux"))]
    // SAFETY: plain libc calls on the calling thread with a properly initialised cpu set.
    unsafe {
        let cores = fast_cores();
        if !cores.is_empty() {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &core in cores {
                libc::CPU_SET(core, &mut set);
            }
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 10);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_fast_cluster() {
        // Exynos 9611 (Galaxy M31): 4 x 2.3 GHz + 4 x 1.7 GHz.
        let m31: Vec<_> = (0..8)
            .map(|c| (c, if c < 4 { 1_704_000 } else { 2_314_000 }))
            .collect();
        assert_eq!(big_cores(&m31), vec![4, 5, 6, 7]);
        // 1 + 4 + 3 layout.
        let tri = [
            (0, 1_800_000),
            (1, 1_800_000),
            (2, 1_800_000),
            (3, 2_600_000),
            (4, 2_600_000),
            (5, 2_600_000),
            (6, 2_600_000),
            (7, 2_800_000),
        ];
        assert_eq!(big_cores(&tri), vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn unknown_frequencies_give_no_answer() {
        assert!(big_cores(&[]).is_empty());
        assert!(big_cores(&[(0, 0), (1, 0)]).is_empty());
    }

    #[test]
    fn default_threads_stay_sensible() {
        let n = default_threads(8);
        assert!((2..=8).contains(&n));
        assert!(default_threads(1) >= 1);
    }
}
