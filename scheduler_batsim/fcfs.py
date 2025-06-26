from batsim_py.simulator import SimulatorHandler

class FCFSScheduler:
    def __init__(self, simulator: SimulatorHandler) -> None:
        self.simulator = simulator

    def __str__(self) -> str:
        return "FCFS"

    def schedule(self) -> None:
        """  First Come First Served policy """
        assert self.simulator.is_running

        # test_time = [
        #     1795, 1945, 2095, 2698, 4771, 5194, 6127, 7467, 9125, 9275, 9574, 10765,
        #     11517, 11964, 12265, 12565, 12867, 13319, 16325, 16468, 16915, 17220, 17369,
        #     18270, 18567, 19173, 19472, 19621, 21124, 22308, 22459, 22759, 22907, 23507,
        #     23660, 26527, 27249, 31142, 32244, 33733, 36158, 36307, 36459, 36759, 36908,
        #     37209, 37209, 37362, 37510, 37811, 38113, 38416, 38416, 38867, 41278, 41580,
        #     42478, 42775, 42775, 42927, 43078, 43078, 43229, 43377, 43377, 44124
        # ]
        # test_time = [
        #     1066, 1519, 1816, 1821, 1971, 2422, 3614, 3614, 3915, 4967, 5273, 8115,
        #     9019, 9169, 9471, 9775, 11257, 11555, 12912, 16975, 17122, 20149, 22252,
        #     23301, 24500, 24512, 25403, 26605, 26755, 26907, 27057, 27354, 29605, 30808,
        #     30808, 31260, 32308, 35613, 37401, 37704, 37851, 38000, 38155, 38445, 38743,
        #     42497, 42648, 43990
        # ]

        test_time = [453]
        
        if self.simulator.current_time in test_time:
            for job in self.simulator.queue:
                print(f"Job {job.id} - res: {job.res}, walltime: {job.walltime}")
            input('Press key to continue')
                
        for job in self.simulator.queue:
            available = self.simulator.platform.get_not_allocated_hosts()
            # if self.simulator.current_time == 12228:
            #     print('current time:', self.simulator.current_time)
            #     print('aval:', available)
            #     print('job head:', self.simulator.queue)
            #     print('job head:', job.id)
            #     input('Press key to continue')
            if job.res <= len(available):
                # if job.id == "w0!87":
                #     print(self.simulator.current_time)
                #     input('Press key to continue')
                # Schedule if the job can start now.
                allocation = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, allocation)
            else:
                # Otherwise, wait for resources.
                break