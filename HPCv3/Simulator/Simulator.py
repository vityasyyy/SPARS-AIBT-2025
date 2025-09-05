import json
import logging

from numpy import nonzero, record
from HPCv3.Simulator.JobsManager import JobsManager
from HPCv3.Simulator.Scheduler import Scheduler
from HPCv3.Simulator.MachineMonitor import Monitor
from HPCv3.Simulator.PlatformControl import PlatformControl
from HPCv3.Utils import log_output

from datetime import datetime

logger = logging.getLogger("runner")


class Simulator:
    def __init__(self, workload_path, platform_path, start_time, algorithm, agent=None, rl=False):
        with open(workload_path, 'r') as file:
            self.workload_info = json.load(file)
        with open(platform_path, 'r') as file:
            self.platform_info = json.load(file)
        self.PlatformControl = PlatformControl(self.platform_info, start_time)
        self.Monitor = Monitor(self.platform_info, start_time)
        self.current_time = start_time
        self.events = []
        self.is_running = False
        self.num_jobs = len(self.workload_info['jobs'])
        self.num_finished_jobs = 0
        self.jobs_manager = JobsManager()
        self.start_time = start_time
        self.scheduler = Scheduler(self.PlatformControl.get_state(
        ), self.jobs_manager.waiting_queue, algorithm, start_time)

        self.rl = rl

        if rl and agent is None:
            raise ValueError(
                "Agent, must be provided if rl=True")

        self.agent = agent

        self.push_event(start_time, {'type': 'simulation_start'})
        for job in self.workload_info['jobs']:
            job['type'] = 'arrival'
            timestamp = job['subtime'] + start_time

            self.push_event(timestamp, job)

    def push_event(self, timestamp, event):
        found = None
        for x in self.events:
            if x['timestamp'] == timestamp:
                found = x
                break
        if found:
            found['events'].append(event)
        else:
            self.events.append({'timestamp': timestamp, 'events': [event]})

        self.events.sort(key=lambda x: x['timestamp'])

    def start_simulator(self):
        self.is_running = True

    def on_finish(self):
        self.is_running = False
        logger.info(f"Simulation finished at time {self.current_time}.")
        self.Monitor.on_finish()
        message = {'now': self.current_time, 'event_list': [
            {'timestamp': self.current_time, 'events': [{'type': 'simulation_finished'}]}]}
        return message

    def proceed(self):
        if len(self.events) == 0:
            message = self.on_finish()
            return message

        self.current_time, events = self.events.pop(0).values()

        for e in events:
            row = [f"[Time={self.current_time:.2f}]"]

            # Always start with job_id and type
            if "job_id" in e:
                row.append(f"job_id={e['job_id']}")
            if "type" in e:
                row.append(f"type={e['type']}")

            # Add remaining keys
            for k, v in e.items():
                if k in ("job_id", "type"):
                    continue
                # Round start_time and subtime
                if k in ("start_time", "subtime") and isinstance(v, (float, int)):
                    v = round(v, 2)
                row.append(f"{k}={v}")

            logger.info(" ".join(row))

        self.Monitor.record(mode='before', current_time=self.current_time)
        event_priority = {
            'turn_on': 0,
            'turn_off': 1,
            'execution_finished': 2,
            'execution_start': 3,
            'arrival': 4,
            'switch_off': 5,
            'switch_on': 6
        }

        events = sorted(events, key=lambda e: event_priority.get(
            e['type'], float('inf')))
        record_job_execution = []
        record_job_arrival = []
        need_rl = False
        for event in events:
            self.event = event
            if self.event['type'] == 'switch_off':
                result_events = self.PlatformControl.switch_off(
                    self.event['nodes'], self.current_time)
                for event_entry in result_events:
                    self.push_event(
                        event_entry['timestamp'], event_entry['event'])

            elif self.event['type'] == 'turn_off':
                self.PlatformControl.turn_off(self.event['nodes'])
                if self.rl:
                    need_rl = True

            elif self.event['type'] == 'switch_on':
                result_events = self.PlatformControl.switch_on(
                    self.event['nodes'], self.current_time)
                for event_entry in result_events:
                    self.push_event(
                        event_entry['timestamp'], event_entry['event'])

            elif self.event['type'] == 'turn_on':
                self.PlatformControl.turn_on(self.event['nodes'])
                if self.rl:
                    need_rl = True

            elif self.event['type'] == 'arrival':
                self.jobs_manager.add_job_to_waiting_queue(self.event)
                record_job_arrival.append(self.event)
                if self.rl:
                    need_rl = True

            elif self.event['type'] == 'execution_start':
                finish_time, event = self.PlatformControl.compute(
                    self.event['nodes'], self.event, self.current_time)
                self.jobs_manager.remove_job_from_waiting_queue(
                    self.event['job_id'], 'execution_start')
                self.push_event(finish_time, event)

            elif self.event['type'] == 'execution_finished':
                self.PlatformControl.release(self.event['nodes'])
                self.num_finished_jobs += 1
                record_job_execution.append(self.event)
                if self.rl:
                    need_rl = True

            elif self.event['type'] == 'change_dvfs_mode':
                event = self.PlatformControl.change_dvfs_mode(
                    self.event['node'], self.event['mode'])

        self.Monitor.record(mode='after', machines=self.PlatformControl.machines,
                            current_time=self.current_time, record_job_execution=record_job_execution, record_job_arrival=record_job_arrival)

        if self.num_finished_jobs == self.num_jobs:
            message = self.on_finish()
            return message

        if need_rl:
            self.push_event(self.current_time, {'type': 'CALL_RL'})
        message = {'timestamp': self.current_time, 'events': events}
        return {'now': self.current_time, 'event_list': [message]}

    def advance(self):
        now = self.current_time

        while self.current_time == now and self.is_running == True:
            self.proceed()

            scheduler_message = self.scheduler.schedule(self.current_time, self.PlatformControl.get_state(
            ), self.jobs_manager.waiting_queue, self.jobs_manager.scheduled_queue)

            for _data in scheduler_message:
                timestamp = _data['timestamp']
                _events = _data['events']
                for event in _events:
                    self.push_event(timestamp, event)

        if self.rl == True:
            rl_action = self.agent.predict()
            # rl_action = self.translate_rl_action(rl_action)
            self.event.push(rl_action)


def run_simulation(simulator, output_folder):
    simulator.start_simulator()
    while simulator.is_running:
        simulator.advance()

    log_output(simulator, output_folder)
    logger.info(f"Simulation completed. Logs saved to: {output_folder}")
