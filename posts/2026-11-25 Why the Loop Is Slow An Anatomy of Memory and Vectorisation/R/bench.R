# Timing helpers. microbenchmark reports nanoseconds. We report the fastest of
# the repeated runs: external noise (interrupts, GC, other processes) can only
# make an operation slower, never faster, so the minimum is the least noisy
# estimate of what the code actually costs.
bench_ns <- function(fun, times = 10, warmup = 2) {
  for (i in seq_len(warmup)) fun()
  t <- microbenchmark::microbenchmark(fun(), times = times)$time
  min(t)
}

# Peak R heap usage while `fun()` runs, in MB.
peak_mb <- function(fun) {
  gc(reset = TRUE, full = TRUE)
  force(fun())
  g <- gc(full = TRUE)
  sum(g[, 6])
}

machine <- function() {
  list(
    r_version = R.version.string,
    platform = R.version$platform,
    cpu = Sys.getenv("PROCESSOR_IDENTIFIER", unset = NA_character_),
    cores = parallel::detectCores(),
    l1_kb = 48, l2_kb = 512, l3_kb = 12288,
    ram_gb = round(as.numeric(system2("powershell",
      c("-NoProfile", "-Command",
        "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
      stdout = TRUE)) / 1024^3, 1)
  )
}
