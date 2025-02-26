I am developing an AI model for job scheduling in a high-performance computing system. The goal is to minimize **waiting time** and **energy consumption** by deciding when to turn nodes **on and off**. However, due to the **event-driven nature** of the system, there are **challenges with feature selection** and **reward function design** that need to be addressed.  

---

### **Key System Characteristics:**  
1. **Event-based simulator**: Actions occur only when events are triggered.  
2. **Multiple events can occur at the same time**: This affects how AI actions influence the system.  
3. **Event types**:  
   - **Turn_on**: Nodes switch on and become available after a transition time. FCFS & backfilling are used to pop jobs from the queue and execute them.  
   - **Turn_off**: Nodes are deallocated and become inactive after a transition time.  
   - **Arrival**: A new job enters the queue.  
   - **Execution_start**: A job starts running.  
   - **Execution_finished**: A job completes, releasing its allocated nodes. FCFS & backfilling immediately pop jobs from the queue to start execution.  
4. **Job execution follows simple FCFS with backfilling**:  
   - No complex priority system.  
   - A job that arrives first has higher priority.  
   - If a later job can fit without delaying the next job, it is scheduled via backfilling.  
5. **AI can only switch specific node states**:  
   - **Nodes that can be turned on**: Sleeping (inactive) nodes.  
   - **Nodes that can be turned off**: Idle nodes.  
   - **Nodes in other states (computing, switching_on, switching_off) should not be included in the AI’s decision-making.**  

---

### **Current Data Available**  

#### **1. Job Queue & Waiting Time Information:**  
- List of jobs in the waiting queue.  
- Total waiting time of jobs still in the queue.  
- Total waiting time, including executed jobs.  

#### **2. Energy Consumption Metrics:**  
- Total energy consumption.  
- Current energy consumption rate based on node states.  

#### **3. System Information:**  
- Requested resources from waiting jobs (derived from the queue).  
- Number of nodes in each state (derived from node states).  

#### **4. Other Potentially Less Critical Data:**  
- List of currently active jobs.  
- List of executed jobs.  

---

### **AI Feature Structure & Issues**  

Currently, the AI receives a **list of node features**, where each node has:  
- **Global features**: e.g., the number of jobs in the waiting queue.  
- **Node-specific features**: e.g., node state and total energy consumed by the node so far.  

#### **Critical Issue with Node Features:**  
- **Only sleeping (inactive) and idle nodes should be considered** for turning on and off.  
- **Nodes in computing, switching_on, or switching_off states should be completely ignored.**  SO FOR FUCK SAKE PLEASE STOP ADDING 2,3,4 AND FOR ANOTHER STATE. LISTEN ONLY SLEEPING AND IDLE NODES WILL BE PASSED INTO PREDICTION.
- **Are there any missing features that could improve AI performance?** Should I prioritize **global** or **node-specific** features?  

---

### **Issues with the Reward Function & Redesign Needs**  

My current reward function is based on a research paper where AI takes action **every 30 minutes**, and the reward is calculated based on:  
- **Reward 1**: Immediate scheduling performance.  
- **Reward 2**: Jobs that arrive in the next 30 minutes but remain unexecuted due to AI decisions.  

However, **this logic is invalid in my event-based system** because:  
- There are **no fixed time intervals** between AI decisions.  
- The next event can occur **immediately after the current event (delta_t = 0)**.  

#### **Event Timing Problem:**  
- **If two events occur at the same time, and the AI's action has no meaningful impact, it can still receive a reward, leading to exploitation.**  
- **Example:** If a node is turned on and then immediately turned off in the next event (same timestamp), the AI learns to spam turning nodes on/off to game the reward function.  

#### **DeepSeek’s Reward Function Issue:**  
- The current method **calculates wasted energy between the current event and the next event**.  
- **Exploit Example:**  
  - Suppose **turn_on** and **turn_off** events occur at the same timestamp.  
  - The AI **turns on nodes**, then immediately **turns them off**.  
  - The simulator copies the state and simulates the next event (turn_off).  
  - Since the time gap is 0, **turning on and off gets a high reward** (low negative value), encouraging AI to spam switching.  
- **At some point, AI starts turning nodes on/off continuously for good rewards.**  

---

### **Reward Should Calculate the Impact on the Next Event**  
- After an event, AI takes an action (turning nodes on/off).  
- A copy of the **entire simulator** is created.  
- The **copied simulator executes one step** to simulate the next event.  
- The **reward is based on the difference between the original and next-state simulator.**  

---

### **Potential Issues & Open Questions**  

1. **How should the AI be penalized for bad decisions?**  
   - Should I introduce a penalty when a job fails execution due to a lack of resources?  
   - Should penalties be applied if AI turns off nodes needed in the next event?  

2. **How many steps should be simulated for reward calculation?**  
   - Should I check only the next event, or should I simulate multiple steps (e.g., 10 future events) to better assess AI impact?  
   - Simulating more steps might give a better reward function but may increase computational cost.  

3. **Should I introduce a job execution penalty?**  
   - If a **job fails to execute** because AI turned off nodes **immediately after execution_finished**, should a penalty be added?  
   - Example:  
     - **Execution_finished releases 8 nodes.**  
     - The simulator **pops 2 jobs (4 nodes each) from the waiting queue** and schedules them as execution_start.  
     - If the AI immediately **turns off all 8 nodes**, the execution_start event **fails**, and jobs return to the waiting queue.  
     - Should there be a **penalty for this failure?**  

4. **How should I prevent AI from exploiting the reward function?**  
   - Adding **delta_t (time gap between events) to the reward** is **not useful** because if delta_t = 0, AI actions at event 1 still get a high reward.  
   - Manually **adding penalties** (e.g., "if job fails execution, apply -X reward") is problematic because **I don’t know the ideal constant value** to use.  

---

### **Final Question:**  
What is the best way to **calculate the reward** based on **AI actions and their impact on the next event**?  
- Should I only look at the **immediate next event**, or should I simulate **multiple future steps**?  
- How do I prevent **reward exploitation** from AI spamming actions like switching nodes on/off?  
- What is the best approach to ensure that **AI learns to optimize waiting time and energy consumption correctly** without being misled by event timing issues?  

Would appreciate insights into **feature selection, reward function redesign, and preventing AI exploitation**.