from .fcfs import FCFSScheduler

class EASYScheduler(FCFSScheduler):
    def __str__(self) -> str:
        return "EASY"

    def schedule(self) -> None:
        super().schedule() # Schedule with FCFS

        # Apply the backfilling mechanism
        if len(self.simulator.queue) >= 2:
            self.backfill()

    def backfill(self) -> None:
        assert len(self.simulator.queue) >= 2
        
        # The priority job is the first job in the queue.
        p_job = self.simulator.queue[0]

        # The remaining jobs can be scheduled if they do not delay p_job.
        backfilling_queue = self.simulator.queue[1:]

        # Get the next expected releases
        next_releases = sorted(self.simulator.agenda, key=lambda a: a.release_time)
        
        #next rilis = [{rilis: 1}, {rilis: 2}, {host: adk, rilis: 3}, {rilis: 4}, {rilis: 5},  ]
        # Get the last required host for p_job.
        # if self.simulator.current_time == 4368:
        #     print(p_job.id)
        #     for r in next_releases:
        #         print(r.host)
        #         print(r.release_time)
        #     input('next_releases')
            
        last_host = next_releases[p_job.res - 1]

        # The last host release time will be the p_job expected start time.
        p_start_t = last_host.release_time

        # Find candidates and reserve resources for p_job.
        candidates = [r.host.id for r in next_releases if r.release_time <= p_start_t]

        # Try to maximize the number of hosts available for the remaining queue.
        reservation = candidates[-p_job.res:]
        test_time = [453]
        
           
        # Let's try to start some jobs earlier.
        for job in backfilling_queue:
            available = self.simulator.platform.get_not_allocated_hosts()  # Hosts
            not_reserved = [h for h in available if h.id not in reservation]
            
            if self.simulator.current_time in test_time and job.id == "w0!11":
                print(p_start_t)
                print(job.res)
                print(not_reserved)
                print(available)
                for r in next_releases:
                    print(r.host.id, r.release_time)
                print(candidates)
                input('Press key to continue')
            if self.simulator.current_time in test_time and job.id == "w0!19":
                print(p_start_t)
                print(job.res)
                print(not_reserved)
                print(available)
                input('Press key to continue')
                
            # if self.simulator.current_time == 313:
            #     print(backfilling_queue)
            #     print(candidates)
            #     print(reservation)
            #     print(p_job)
            #     print(p_job.res)
            #     print(job.res)
            #     print(not_reserved)
            #     print(job.walltime)
            #     print(p_start_t)
            #     print(available)
            #     input('Press key to continue')
                
            #     input('x in bf')
            
            if job.res <= len(not_reserved):
                # Schedule job on not reserved hosts.
                # if job.id == "w0!87":
                #     print(self.simulator.current_time)
                #     input('Press key to continue')
                allocation = [h.id for h in not_reserved[:job.res]]
                self.simulator.allocate(job.id, allocation)
            elif job.walltime and job.walltime <= p_start_t and job.res <= len(available):
                # Schedule job on reserved hosts without delaying p_job.
                # if job.id == "w0!87":
                #     print(self.simulator.current_time)
                #     input('Press key to continue')
                allocation = [h.id for h in available[:job.res]]
                self.simulator.allocate(job.id, allocation)