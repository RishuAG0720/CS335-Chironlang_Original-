# # #!/usr/bin/python3
# # # -*- coding: utf-8 -*-
# # # Repeat counter handling for ChironLang

# # from ChironAST.ChironAST import Var, AssignmentCommand

# # class RepeatCounterHandler:
# #     def __init__(self, cfg):
# #         self.cfg = cfg
# #         self.repeat_counters = set()
# #         self.base_variables = {}
# #         self.versions = {}  # Track versions for each repeat counter
    
# #     def analyze_repeat_counters(self):
# #         """Analyze repeat counter variables in the CFG"""
# #         for block in self.cfg.nodes():
# #             for instr, _ in block.instrlist:
# #                 if isinstance(instr, AssignmentCommand):
# #                     var_name = instr.lvar.varname
# #                     if var_name.startswith('__rep_counter_'):
# #                         self.repeat_counters.add(var_name)
# #                         base_name = var_name.split('_')[2]
# #                         self.base_variables[var_name] = base_name
# #                         self.versions[var_name] = 0  # Initialize version counter
    
# #     def initialize_repeat_counters(self):
# #         """Initialize repeat counter variables in the START block"""
# #         start_block = None
# #         for block in self.cfg.nodes():
# #             if block.name == "START":
# #                 start_block = block
# #                 break
        
# #         if start_block:
# #             for counter in self.repeat_counters:
# #                 base_name = self.base_variables[counter]
# #                 # Find the base variable's value in the START block
# #                 base_value = None
# #                 for instr, _ in start_block.instrlist:
# #                     if isinstance(instr, AssignmentCommand) and instr.lvar.varname == base_name:
# #                         base_value = instr.rexpr
# #                         break
                
# #                 if base_value is not None:
# #                     # Create a new version of the repeat counter
# #                     self.versions[counter] += 1
# #                     new_counter = f"{counter}_{self.versions[counter]}"
# #                     instr = AssignmentCommand(Var(new_counter), base_value)
# #                     start_block.instrlist.append((instr, -1))
# #                 else:
# #                     # If base variable not found, initialize with 0
# #                     self.versions[counter] += 1
# #                     new_counter = f"{counter}_{self.versions[counter]}"
# #                     instr = AssignmentCommand(Var(new_counter), Var("0"))
# #                     start_block.instrlist.append((instr, -1))
    
# #     def get_base_variable(self, counter_name):
# #         """Get the base variable name for a repeat counter"""
# #         return self.base_variables.get(counter_name)
    
# #     def is_repeat_counter(self, var_name):
# #         """Check if a variable is a repeat counter"""
# #         return var_name in self.repeat_counters
    
# #     def get_current_version(self, counter_name):
# #         """Get the current version number for a repeat counter"""
# #         return self.versions.get(counter_name, 0)
    
# #     def increment_version(self, counter_name):
# #         """Increment the version number for a repeat counter"""
# #         if counter_name in self.versions:
# #             self.versions[counter_name] += 1
# #             return self.versions[counter_name]
# #         return 0 

# #!/usr/bin/python3
# # -*- coding: utf-8 -*-
# # Repeat counter handling for ChironLang

# from ChironAST.ChironAST import Var, AssignmentCommand

# class RepeatCounterHandler:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.repeat_counters = set()
#         self.base_variables = {}
#         self.versions = {}  # Track versions for each repeat counter

#     def analyze_repeat_counters(self):
#         """Analyze repeat counter variables in the CFG"""
#         for block in self.cfg.nodes():
#             for instr, _ in block.instrlist:
#                 if isinstance(instr, AssignmentCommand):
#                     var_name = instr.lvar.varname
#                     if var_name.startswith('__rep_counter_'):
#                         self.repeat_counters.add(var_name)
#                         base_name = var_name.split('_')[2]
#                         self.base_variables[var_name] = base_name
#                         self.versions[var_name] = 0  # Initialize version counter

#     def initialize_repeat_counters(self):
#         """Initialize repeat counter variables in the START block"""
#         start_block = None
#         for block in self.cfg.nodes():
#             if block.name == "START":
#                 start_block = block
#                 break

#         if start_block:
#             for counter in self.repeat_counters:
#                 base_name = self.base_variables[counter]
#                 base_value = None
#                 for instr, _ in start_block.instrlist:
#                     if isinstance(instr, AssignmentCommand) and instr.lvar.varname == base_name:
#                         base_value = instr.rexpr
#                         break

#                 self.versions[counter] += 1
#                 new_counter = f"{counter}_{self.versions[counter]}"
#                 if base_value is not None:
#                     instr = AssignmentCommand(Var(new_counter), base_value)
#                 else:
#                     instr = AssignmentCommand(Var(new_counter), Var("0"))
#                 start_block.instrlist.append((instr, -1))

#     def get_base_variable(self, counter_name):
#         """Get the base variable name for a repeat counter"""
#         return self.base_variables.get(counter_name)

#     def is_repeat_counter(self, var_name):
#         """Check if a variable is a repeat counter"""
#         return any(var_name.startswith(counter) for counter in self.repeat_counters)

#     def get_current_version(self, counter_name):
#         """Get the current version number for a repeat counter"""
#         for counter in self.versions:
#             if counter_name.startswith(counter):
#                 return self.versions[counter]
#         return 0

#     def increment_version(self, counter_name):
#         """Increment the version number for a repeat counter"""
#         for counter in self.versions:
#             if counter_name.startswith(counter):
#                 self.versions[counter] += 1
#                 return self.versions[counter]
#         return 0



#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Repeat counter handling for ChironLang

from ChironAST.ChironAST import Var, AssignmentCommand

class RepeatCounterHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.repeat_counters = set()
        self.base_variables = {}
        self.versions = {}  # Track versions for each repeat counter

    def analyze_repeat_counters(self):
        """Analyze repeat counter variables in the CFG"""
        for block in self.cfg.nodes():
            for instr, _ in block.instrlist:
                if isinstance(instr, AssignmentCommand):
                    var_name = instr.lvar.varname
                    if var_name.startswith('__rep_counter_'):
                        base_name = '_'.join(var_name.split('_')[:3])
                        self.repeat_counters.add(base_name)
                        self.base_variables[base_name] = var_name.split('_')[2]
                        self.versions[base_name] = 0

    def initialize_repeat_counters(self):
        """Initialize repeat counter variables in the START block"""
        start_block = None
        for block in self.cfg.nodes():
            if block.name == "START":
                start_block = block
                break

        if start_block:
            for counter in self.repeat_counters:
                base_name = self.base_variables[counter]
                base_value = None
                for instr, _ in start_block.instrlist:
                    if isinstance(instr, AssignmentCommand) and instr.lvar.varname == base_name:
                        base_value = instr.rexpr
                        break

                self.versions[counter] += 1
                new_counter = f"{counter}_{self.versions[counter]}"
                if base_value is not None:
                    instr = AssignmentCommand(Var(new_counter), base_value)
                else:
                    instr = AssignmentCommand(Var(new_counter), Var("0"))
                start_block.instrlist.append((instr, -1))

    def get_base_variable(self, counter_name):
        """Get the base variable name for a repeat counter"""
        for base in self.base_variables:
            if counter_name.startswith(base):
                return self.base_variables[base]
        return None

    def is_repeat_counter(self, var_name):
        """Check if a variable is a repeat counter (handles versioned forms)"""
        return any(var_name.startswith(counter) for counter in self.repeat_counters)

    def get_current_version(self, counter_name):
        """Get the current version number for a repeat counter (from base)"""
        for base in self.versions:
            if counter_name.startswith(base):
                return self.versions[base]
        return 0

    def increment_version(self, counter_name):
        """Increment and return new version for a repeat counter"""
        for base in self.versions:
            if counter_name.startswith(base):
                self.versions[base] += 1
                return self.versions[base]
        return 0
