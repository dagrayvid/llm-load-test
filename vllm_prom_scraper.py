import requests
import logging
import time

# Define the function to run in its own process
def query_metrics(logger_q, log_level, host, shared_list, metrics_port_and_endpoint):
    """
    Queries Prometheus metrics from multiple IPs and stores them in a shared dictionary.

    Args:
    - args
    - shared_dict (multiprocessing.Manager().dict()): Shared dictionary to store metrics.
    - metrics_port_and_endpoint (str): Port and endpoint for metrics (e.g., "9090/metrics").
    """

    """Init logging."""
    qh = logging.handlers.QueueHandler(logger_q)
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(qh)
    logger = logging.getLogger("metrics")

    import socket
    ip_list = []
    ais = socket.getaddrinfo(host,0,0,0,0)
    for result in ais:
        ip_list.append(result[-1][0])
        ip_list = list(set(ip_list))
    logger.info("Hosts: %s", ip_list)

    while True:
        list_of_dicts = []
  # Run indefinitely until process is terminated
        for ip in ip_list:
            ip_metrics = {"ip": ip}
            try:
                r = requests.get(f"http://{ip}:{metrics_port_and_endpoint}", timeout=0.2)
                for line in r.iter_lines():
                    if b"vllm:gpu_cache_usage_perc{" in line:
                        kv_cache_pct = line.decode("utf-8").split(" ")[-1]
                        ip_metrics["kv_cache_pct"] = float(kv_cache_pct)
                    if b"vllm:num_requests_waiting{" in line:
                        num_requests_waiting = line.decode("utf-8").split(" ")[-1]
                        ip_metrics["num_requests_waiting"] = float(num_requests_waiting)
                list_of_dicts.append(ip_metrics)
            except requests.exceptions.Timeout:
                logger.warning(f"Warning: timed out on pod {ip}")
            
        sorted_metrics = sorted(list_of_dicts, key=lambda x: (x['num_requests_waiting'], x['kv_cache_pct']))
        logger.info(f"Sorted metrics: {sorted_metrics}")

        num_return = int(len(sorted_metrics) / 2)
        #shared_list[:] = [entry["ip"] for entry in sorted_metrics[:num_return]] # shared_list +
        shared_list[:] = [entry["ip"] for entry in sorted_metrics[:]] # shared_list +
        logger.info(f"Shared list: {shared_list}")

        time.sleep(0.05)  # Adjust the sleep time as needed

