Multi-Agent Simulation in Robotic Mobile Fulfillment System

Description: 
That's a multi-agent simulation system for robotic mobile fulfillment system. I want to implement it in Python. Any module in system I hope it can have a template class file as an interface. And also I hope the inital configuration could follow the .json file in Config folder (including adjusting the parameters of maps and robots in .json file).


Expected Architecture (if you think something is unnecessary, please remove it):
- Config
- Engine
- Policies
  - OrderGenerator
  - TaskAssigner
  - PathPlanner 
  - Scheduler (optional)
- Visualization
  - UI
- WorldState
  - AgentState 
  - OrderState
  - TaskState 
  - MapState
  - PobState 
- Debug
  - Logger

The Visualization module is no need to write in hurry, it can be written later. Let all the backend processes run smoothly without any problems first.


PS: 
Don't need to follow existing MAPF convention from CBS_sim project!!!
I want the simulation which can run continously instead of being interrupted when a certain number of ticks are reached. And the default path planning algorithm can choose A star. For other algorithms, you can choose a popular and common one.
