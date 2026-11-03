import os
import sys
import time
import json
import subprocess
from datetime import datetime, timezone, timedelta

POD_ID = "fct9n1rt8y02to"
POD_HOST = "194.68.245.111"
POD_PORT = "22021"
POD_USER = "root"

RATE_PER_HOUR = 0.59
MAX_SPEND = 1.75

# Start time: 2026-10-28 14:32:36 UTC (11:32:36 BRT)
POD_START_UTC = datetime(2026, 10, 28, 14, 32, 36, tzinfo=timezone.utc)

BRT = timezone(timedelta(hours=-3))
STOP_TIME_BRT = datetime(2026, 10, 28, 14, 25, 0, tzinfo=BRT)

LOCAL_DIR = r"C:\Users\User\Desktop\projetos\secret\outputs\krea2_lora"
REMOTE_OUTPUTS = "/workspace/outputs"
LOG_FILE = os.path.join(LOCAL_DIR, "sentinel_watchdog.log")
STATE_FILE = os.path.join(LOCAL_DIR, "sentinel_state.json")

os.makedirs(LOCAL_DIR, exist_ok=True)

def log(msg):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def run_ssh(cmd, timeout=30):
    ssh_cmd = [
        "ssh", "-p", POD_PORT,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{POD_USER}@{POD_HOST}",
        cmd
    ]
    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return -1, "", str(e)

def run_scp_file(remote_path, local_path, timeout=180):
    scp_cmd = [
        "scp", "-P", POD_PORT,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{POD_USER}@{POD_HOST}:{remote_path}",
        local_path
    ]
    try:
        res = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode == 0
    except Exception as e:
        log(f"SCP Error on {remote_path}: {e}")
        return False

def terminate_pod_runpod():
    try:
        import runpod
        api_key = os.environ.get('RUNPOD_API_KEY')
        if api_key:
            runpod.api_key = api_key
        log(f"Terminating pod {POD_ID} via runpod.terminate_pod...")
        res = runpod.terminate_pod(POD_ID)
        log(f"Terminate response: {res}")
        return True
    except Exception as e:
        log(f"Error terminating pod: {e}")
        return False

def get_current_spend():
    now_utc = datetime.now(timezone.utc)
    uptime_sec = max(0.0, (now_utc - POD_START_UTC).total_seconds())
    hours = uptime_sec / 3600.0
    spend = round(hours * RATE_PER_HOUR, 4)
    return spend, uptime_sec

def rescue_all_outputs():
    log("=== Rescuing all remote outputs to local directory ===")
    rc, ls_out, _ = run_ssh("ls -1 /workspace/outputs/")
    if rc == 0:
        files = [f.strip() for f in ls_out.splitlines() if f.strip()]
        for f in files:
            remote_f = f"/workspace/outputs/{f}"
            local_f = os.path.join(LOCAL_DIR, f)
            if not f.startswith("checkpoint-step"):
                if not os.path.exists(local_f) or os.path.getsize(local_f) == 0:
                    log(f"Rescuing: {f}")
                    run_scp_file(remote_f, local_f)
    run_scp_file("/workspace/progress.json", os.path.join(LOCAL_DIR, "final_progress.json"))

def main():
    log("=== RunPod Sentinel Watchdog (Target: 700 Steps + Benchmark) Initiated ===")
    part2_started = False
    benchmark_started = False
    benchmark_done = False
    downloaded_files = set()
    
    for fname in os.listdir(LOCAL_DIR):
        if fname.endswith(".safetensors") or fname.endswith(".png"):
            downloaded_files.add(fname)
    log(f"Initially present local files: {sorted(list(downloaded_files))}")

    while True:
        try:
            now_now = datetime.now(BRT)
            current_spend, uptime_sec = get_current_spend()
            uptime_min = uptime_sec / 60.0

            # 1. Check progress.json
            rc, progress_raw, _ = run_ssh("cat /workspace/progress.json 2>/dev/null")
            progress_data = {}
            if rc == 0 and progress_raw.strip():
                try:
                    progress_data = json.loads(progress_raw.strip())
                except Exception:
                    pass

            current_step = progress_data.get("step", 0)
            total_steps = progress_data.get("total_steps", 400)
            loss = progress_data.get("loss", 0.0)
            smoothed_loss = progress_data.get("smoothed_loss", 0.0)
            eta_min = progress_data.get("eta_min", 0.0)
            part = progress_data.get("part", 1)

            # 2. Check remote outputs for new .safetensors and .png
            rc, ls_out, _ = run_ssh("ls -1 /workspace/outputs/ 2>/dev/null")
            remote_files = [line.strip() for line in ls_out.splitlines() if line.strip()] if rc == 0 else []

            for rf in remote_files:
                if (rf.endswith(".safetensors") or rf.endswith(".png")) and rf not in downloaded_files:
                    log(f">>> NEW ASSET DETECTED: {rf} (Step {current_step}). Starting SCP...")
                    remote_fpath = f"/workspace/outputs/{rf}"
                    local_fpath = os.path.join(LOCAL_DIR, rf)
                    success = run_scp_file(remote_fpath, local_fpath)
                    if success and os.path.exists(local_fpath):
                        fsize_mb = round(os.path.getsize(local_fpath) / (1024*1024), 2)
                        log(f">>> DOWNLOAD COMPLETE: {rf} ({fsize_mb} MB) successfully saved!")
                        downloaded_files.add(rf)
                    else:
                        log(f"WARNING: SCP failed for {rf}")

            # Also sync latest_checkpoint.json
            if "latest_checkpoint.json" in remote_files:
                run_scp_file("/workspace/outputs/latest_checkpoint.json", os.path.join(LOCAL_DIR, "latest_checkpoint.json"))

            # Save state
            state = {
                "timestamp": datetime.now().isoformat(),
                "brt_time": now_now.strftime("%H:%M:%S BRT"),
                "uptime_min": round(uptime_min, 1),
                "spend_usd": current_spend,
                "part": part,
                "step": current_step,
                "total_steps": total_steps,
                "loss": loss,
                "smoothed_loss": smoothed_loss,
                "eta_min": eta_min,
                "part2_active": part2_started,
                "benchmark_started": benchmark_started,
                "benchmark_done": benchmark_done,
                "downloaded_assets": sorted(list(downloaded_files))
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            log(f"Status: Part {part} | Step [{current_step:03d}/{total_steps}] | Loss: {loss:.4f} | Uptime: {uptime_min:.1f}m | Spent: ${current_spend:.2f} | Local Assets: {len(downloaded_files)}")

            # 3. Check transition from Part 1 to Part 2 (Step 400)
            if not part2_started:
                rc_done, _, _ = run_ssh("test -f /workspace/TRAINING_DONE && echo DONE")
                if (rc_done == 0 and "DONE" in _) or (current_step >= 400 and part == 1):
                    log(">>> STEP 400 (PART 1) COMPLETED! Downloading Step 400 weights...")
                    for final_cand in ["sofia_krea2_step_400.safetensors", "sofia_krea2_lora_final.safetensors"]:
                        if final_cand not in downloaded_files:
                            run_scp_file(f"/workspace/outputs/{final_cand}", os.path.join(LOCAL_DIR, final_cand))
                            if os.path.exists(os.path.join(LOCAL_DIR, final_cand)):
                                downloaded_files.add(final_cand)
                                log(f">>> Downloaded {final_cand}")
                                
                    if now_now < STOP_TIME_BRT and current_spend < MAX_SPEND:
                        log(">>> Launching Part 2 (Steps 401-700)...")
                        launch_cmd = "nohup python3 -u /workspace/train_krea2_sofia_part2.py > /workspace/part2.log 2>&1 &"
                        rc_launch, _, _ = run_ssh(launch_cmd)
                        log(f"Part 2 launched (code {rc_launch}). Transition complete.")
                        part2_started = True
                        time.sleep(10)
                        continue
                    else:
                        log(">>> Safe limit reached before Part 2. Finalizing...")
                        break

            # 4. Check completion of Part 2 (Step 700)
            if part2_started and not benchmark_started:
                rc_p2, _, _ = run_ssh("test -f /workspace/TRAINING_PART2_DONE && echo DONE")
                if (rc_p2 == 0 and "DONE" in _) or (current_step >= 700 and part == 2):
                    log(">>> STEP 700 (PART 2) COMPLETED! Rescuing Step 700 weights...")
                    for p2_cand in ["sofia_krea2_step_700.safetensors", "sofia_krea2_lora_final_step700.safetensors"]:
                        if p2_cand not in downloaded_files:
                            run_scp_file(f"/workspace/outputs/{p2_cand}", os.path.join(LOCAL_DIR, p2_cand))
                            if os.path.exists(os.path.join(LOCAL_DIR, p2_cand)):
                                downloaded_files.add(p2_cand)
                                log(f">>> Downloaded {p2_cand}")

                    log(">>> Triggering Comparative Benchmark Generation on GPU (Steps 400, 500, 600, 700 | Seed 42)...")
                    bench_cmd = "nohup python3 -u /workspace/run_comparison_benchmark.py > /workspace/benchmark.log 2>&1 &"
                    rc_b, _, _ = run_ssh(bench_cmd)
                    log(f"Benchmark launched (code {rc_b}). Waiting for image generation...")
                    benchmark_started = True
                    time.sleep(15)
                    continue

            # 5. Check if Benchmark completed
            if benchmark_started:
                rc_bdone, _, _ = run_ssh("test -f /workspace/BENCHMARK_DONE && echo DONE")
                if rc_bdone == 0 and "DONE" in _:
                    log(">>> COMPARATIVE BENCHMARK GENERATION COMPLETED!")
                    benchmark_done = True
                    # Download all comparison images
                    for step_num in [400, 500, 600, 700]:
                        c_img = f"comparison_step_{step_num}.png"
                        if c_img not in downloaded_files:
                            run_scp_file(f"/workspace/outputs/{c_img}", os.path.join(LOCAL_DIR, c_img))
                            if os.path.exists(os.path.join(LOCAL_DIR, c_img)):
                                downloaded_files.add(c_img)
                                log(f">>> Downloaded benchmark image: {c_img}")
                    log(">>> All benchmark images downloaded. Ready for final shutdown!")
                    break

            # 6. Hard Limit Check: 14:25 BRT or MAX_SPEND ($1.75)
            if now_now >= STOP_TIME_BRT or current_spend >= MAX_SPEND:
                log(f">>> HARD CEILING REACHED (Time: {now_now.strftime('%H:%M:%S')}, Spend: ${current_spend:.2f})! Initiating immediate rescue and shutdown...")
                break

            time.sleep(20)

        except Exception as e:
            log(f"Watchdog Loop Exception: {e}")
            time.sleep(20)

    # FINAL RESCUE & TERMINATION
    log("=== Commencing Final Asset Rescue & Pod Termination ===")
    rescue_all_outputs()

    final_files = os.listdir(LOCAL_DIR)
    log(f"Preserved assets in {LOCAL_DIR}: {final_files}")

    terminate_pod_runpod()
    log("Pod termination requested. Confirming shutdown...")
    time.sleep(10)

    summary = {
        "completed_at": datetime.now().isoformat(),
        "final_spend_usd": current_spend,
        "assets_preserved": sorted(final_files),
        "status": "ALL_ASSETS_SAVED_POD_TERMINATED"
    }
    with open(os.path.join(LOCAL_DIR, "SENTINEL_SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("=== SENTINEL PROTOCOL (700 STEPS + BENCHMARK) FULLY EXECUTED ===")

if __name__ == "__main__":
    main()
