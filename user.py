"""User definition."""

import logging
import queue
import time


class User:
    """Define a user."""

    def __init__(
        self,
        user_id,
        dataset,
        schedule_q,
        stop_q,
        results_pipe,
        plugin,
        logger_q,
        log_level,
        run_duration,
        rate_limited,
    ):
        """Initialize object."""
        self.user_id = user_id
        self.plugin = plugin
        self.dataset = dataset
        self.dataset_idx = 0
        self.schedule_q = schedule_q
        self.stop_q = stop_q
        self.results_list = []
        self.results_pipe = results_pipe
        self.logger_q = logger_q
        self.log_level = log_level
        # Must get reset in user process to use the logger created in _init_user_process_logging
        self.logger = logging.getLogger("user")
        self.run_duration = run_duration
        self.rate_limited = rate_limited

    def make_request(self, test_end_time=0, req_schedule_time=None):
        """Make a request."""
        query = self.dataset[self.dataset_idx]
        self.dataset_idx = (self.dataset_idx + 1) % len(self.dataset)

        self.logger.info("User %s making request", self.user_id)
        result = self.plugin.request_func(query, self.user_id, test_end_time)

        if req_schedule_time:
            result.scheduled_start_time = req_schedule_time
        
        result.calculate_results()

        return result

    def _init_user_process_logging(self):
        """Init logging."""
        qh = logging.handlers.QueueHandler(self.logger_q)
        root = logging.getLogger()
        root.setLevel(self.log_level)
        root.handlers.clear()
        root.addHandler(qh)

        self.logger = logging.getLogger("user")
        return logging.getLogger("user")

    def run_user_process(self):
        """Run a process."""
        self._init_user_process_logging()

        # Waits for all processes to actually be started
        while not self.rate_limited and self.schedule_q.empty():
            time.sleep(0.1)
        
        test_end_time = time.time() + self.run_duration
        self.logger.info("User %s starting request loop", self.user_id)

        while self.stop_q.empty():
            try:
                req_schedule_time = self.schedule_q.get(timeout=2)
                if not self.stop_q.empty():
                    break
            except queue.Empty:
                # if timeout passes, queue.Empty will be thrown
                # User should check if stop_q has been set, else poll again
                # self.debug.info("User waiting for a request to be scheduled")
                continue

            result = self.make_request(test_end_time, req_schedule_time=req_schedule_time)


            if result is not None:
                self.results_list.append(result)
            else:
                self.logger.info("Unexpected None result from User.make_request()")

        self.results_pipe.send(self.results_list)

        time.sleep(4)
        self.logger.info("User %s done", self.user_id)
