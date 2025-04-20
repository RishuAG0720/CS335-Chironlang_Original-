#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Static Single Assignment implementation for ChironLang

import sys
import copy
from collections import defaultdict, deque
import uuid

from ChironAST.ChironAST import Var, PhiFunction, AssignmentCommand, ConditionCommand, GotoCommand
from cfg.ChironCFG import BasicBlock, ChironCFG
from repeat_counter import RepeatCounterHandler

class SSATransformer:
    """
    Transforms a Control Flow Graph into Static Single Assignment form.
    
    The algorithm has two main phases:
    1. Insert Phi functions at the beginning of selected blocks
    2. Rename variables (converting each definition to a new version)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.dominance_frontiers = None
        self.dom_tree = None
        self.variable_definitions = defaultdict(list)
        self.variable_uses = defaultdict(set)
        self.counter = defaultdict(int)  # Counters for variable versions
        self.stack = defaultdict(list)   # Stack for renaming variables
        self.loop_headers = set()        # Set of loop header blocks
        self.loop_latches = set()        # Set of loop latch blocks (end of loop)
        self.repeat_handler = RepeatCounterHandler(cfg)  # Add repeat counter handler
        
    def compute_dominators(self):
        """
        Compute dominators for each node in the CFG
        Returns a dictionary mapping nodes to their dominators
        """
        graph = self.cfg.nxgraph
        nodes = list(graph.nodes())
        
        entry = None
        for node in nodes:
            if node.name == "START":
                entry = node
                break
        
        if not entry:
            raise ValueError("No entry node found in the CFG")
        
        dominators = {node: set(nodes) for node in nodes}
        dominators[entry] = {entry}
        
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if node == entry:
                    continue
                
                preds = list(self.cfg.predecessors(node))
                if not preds:
                    continue
                
                new_dom = set(dominators[preds[0]])
                for pred in preds[1:]:
                    new_dom &= dominators[pred]
                
                new_dom = {node} | new_dom
                
                if new_dom != dominators[node]:
                    dominators[node] = new_dom
                    changed = True
        
        return dominators
    
    def build_dominance_tree(self, dominators):
        """
        Build a dominance tree from the dominators dictionary
        Returns a dictionary mapping nodes to their immediate dominator
        """
        idom = {}
        for node in dominators:
            strict_dominators = dominators[node] - {node}
            if not strict_dominators:
                continue
                
            immediate_dom = None
            for dom_candidate in strict_dominators:
                is_immediate = True
                for other_dom in strict_dominators:
                    if dom_candidate != other_dom and dom_candidate in dominators[other_dom]:
                        is_immediate = False
                        break
                if is_immediate:
                    immediate_dom = dom_candidate
                    break
            
            if immediate_dom:
                idom[node] = immediate_dom
        
        return idom
    
    def compute_dominance_frontiers(self, idom):
        """
        Compute dominance frontiers for each node
        A node's dominance frontier is the set of nodes where its dominance ends
        """
        df = defaultdict(set)
        nodes = list(self.cfg.nxgraph.nodes())
        
        for node in nodes:
            preds = list(self.cfg.predecessors(node))
            if len(preds) < 2:
                continue
            
            for pred in preds:
                runner = pred
                while runner and runner != idom.get(node):
                    df[runner].add(node)
                    runner = idom.get(runner)
        
        return df
    
    def identify_loops(self):
        """
        Identify loops in the CFG (using back edges)
        """
        dominators = self.compute_dominators()
        
        for node in self.cfg.nodes():
            for succ in self.cfg.successors(node):
                if succ in dominators[node]:
                    succ.name = "LOOP_HEADER"
                    node.name = "LOOP_LATCH"
                    self.loop_headers.add(succ)
                    self.loop_latches.add(node)
                    
                    for n in self.cfg.nodes():
                        if succ in dominators[n]:
                            for s in self.cfg.successors(n):
                                if succ not in dominators[s]:
                                    s.name = "AFTER_LOOP"
    
    def analyze_variables(self):
        """
        Analyze variable definitions and uses in each basic block
        """
        for bb in self.cfg.nodes():
            for instr, _ in bb.instrlist:
                if isinstance(instr, AssignmentCommand):
                    var_name = instr.lvar.varname
                    self.variable_definitions[var_name].append(bb)
                    self._find_variable_uses(instr.rexpr, bb)
                elif hasattr(instr, 'expr'):
                    self._find_variable_uses(instr.expr, bb)
                elif hasattr(instr, 'cond'):
                    self._find_variable_uses(instr.cond, bb)
        
        start_block = None
        for bb in self.cfg.nodes():
            if bb.name == "START":
                start_block = bb
                break
        
        if start_block:
            all_vars = set(self.variable_definitions.keys()) | set(self.variable_uses.keys())
            
            for var_name in all_vars:
                if var_name.startswith('__') and not var_name.startswith('__rep_counter_'):
                    continue
                
                if (var_name in self.variable_uses and 
                    not any(block.name == "START" for block in self.variable_definitions[var_name])):
                    
                    needs_initialization = False
                    for bb in self.cfg.nodes():
                        if var_name in self.variable_uses and bb in self.variable_uses[var_name]:
                            if not self._any_definition_reaches(var_name, bb):
                                needs_initialization = True
                                break
                    
                    if needs_initialization and var_name not in [i[0].lvar.varname for i in start_block.instrlist]:
                        target_var = Var(var_name)
                        instr = AssignmentCommand(target_var, Var("0"))
                        start_block.instrlist.append((instr, -1))
                        self.variable_definitions[var_name].append(start_block)
            
            for var_name in list(self.variable_uses.keys()):
                if var_name.startswith('__rep_counter_'):
                    base_name = var_name.split('_')[2]
                    if base_name in self.variable_definitions:
                        target_var = Var(var_name)
                        base_value = None
                        for instr, _ in start_block.instrlist:
                            if isinstance(instr, AssignmentCommand) and instr.lvar.varname == base_name:
                                base_value = instr.rexpr
                                break
                        
                        if base_value is not None:
                            instr = AssignmentCommand(target_var, base_value)
                            start_block.instrlist.append((instr, -1))
                            self.variable_definitions[var_name].append(start_block)
                        else:
                            instr = AssignmentCommand(target_var, Var("0"))
                            start_block.instrlist.append((instr, -1))
                            self.variable_definitions[var_name].append(start_block)

    def _find_variable_uses(self, expr, bb):
        """
        Find variable uses in an expression
        """
        if isinstance(expr, Var):
            var_name = expr.varname
            self.variable_uses[var_name].add(bb)
        elif hasattr(expr, 'lexpr') and hasattr(expr, 'rexpr'):
            self._find_variable_uses(expr.lexpr, bb)
            self._find_variable_uses(expr.rexpr, bb)
        elif hasattr(expr, 'expr'):
            self._find_variable_uses(expr.expr, bb)
    
    def _any_definition_reaches(self, var_name, block):
        """
        Check if any definition of the variable reaches this block
        """
        visited = set()
        return self._definition_reaches_block(var_name, block, visited)

    def _definition_reaches_block(self, var_name, block, visited):
        """
        Check if a definition of the variable reaches this block
        """
        if block in visited:
            return False
        
        visited.add(block)
        
        for instr, _ in block.instrlist:
            if isinstance(instr, AssignmentCommand) and instr.lvar.varname == var_name:
                return True
        
        preds = list(self.cfg.predecessors(block))
        for pred in preds:
            if self._definition_reaches_block(var_name, pred, visited.copy()):
                return True
        
        return False
    
    def place_phi_functions(self):
        """
        Place Phi functions at join points where variables have multiple definitions
        """
        phi_placements = defaultdict(set)
        
        for var_name, def_blocks in self.variable_definitions.items():
            if len(def_blocks) <= 1:
                continue
            
            work_list = list(def_blocks)
            processed = set()
            
            while work_list:
                block = work_list.pop(0)
                if block in processed:
                    continue
                
                processed.add(block)
                
                for frontier_block in self.dominance_frontiers.get(block, set()):
                    if var_name not in phi_placements[frontier_block]:
                        phi_placements[frontier_block].add(var_name)
                        
                        if frontier_block in def_blocks:
                            work_list.append(frontier_block)
        
        for header in self.loop_headers:
            for var_name in self.variable_definitions:
                if var_name.startswith('__') and not var_name.startswith('__rep_counter_'):
                    continue
                    
                is_modified_in_loop = False
                for block in self.variable_definitions[var_name]:
                    if header in self.dominators.get(block, set()):
                        is_modified_in_loop = True
                        break
                        
                if is_modified_in_loop:
                    phi_placements[header].add(var_name)

        for header in self.loop_headers:
            exit_blocks = set()
            loop_blocks = set()
            for block in self.cfg.nodes():
                if header in self.dominators.get(block, set()):
                    loop_blocks.add(block)
                    
            for block in loop_blocks:
                for succ in self.cfg.successors(block):
                    if header not in self.dominators.get(succ, set()):
                        exit_blocks.add(succ)
                        succ.name = "AFTER_LOOP"
            
            for exit_block in exit_blocks:
                for var_name in self.variable_definitions:
                    if var_name.startswith('__') and not var_name.startswith('__rep_counter_'):
                        continue
                        
                    is_modified_in_loop = False
                    for block in self.variable_definitions[var_name]:
                        if header in self.dominators.get(block, set()):
                            is_modified_in_loop = True
                            break
                            
                    if is_modified_in_loop:
                        phi_placements[exit_block].add(var_name)
        
        for block, var_names in phi_placements.items():
            if not hasattr(block, 'phi_functions'):
                block.phi_functions = []
            
            for var_name in var_names:
                target_var = Var(var_name)
                phi_func = PhiFunction(target_var, [])
                block.phi_functions.append(phi_func)
    
    def rename_variables(self):
        """
        Rename variables to ensure each variable is assigned only once
        Uses a recursive algorithm based on dominance tree
        """
        # Get the entry node
        entry = None
        for node in self.cfg.nodes():
            if node.name == "START":
                entry = node
                break
            
        if not entry:
            raise ValueError("No entry node found in the CFG")
            
        # Compute children in dominance tree
        dom_children = defaultdict(list)
        for node, idom_node in self.dom_tree.items():
            dom_children[idom_node].append(node)
            
        # Start renaming from the entry node
        self._rename_variables_recursive(entry, dom_children)
            
        # Initialize variables that are used but not defined
        for var_name in self.variable_uses:
            if var_name not in self.counter:
                self._new_variable_version(var_name)

    def _rename_variables_recursive(self, block, dom_children):
        """
        Recursive function to rename variables in a block and its dominator children
        """
        # Process phi functions first
        for phi in getattr(block, 'phi_functions', []):
            var_name = phi.target_var.varname
            new_version = self._new_variable_version(var_name)
            phi.target_var = Var(f"{var_name}_{new_version}")
            if self.repeat_handler.is_repeat_counter(var_name):
                self.repeat_handler.increment_version(var_name)
        
        # Process normal instructions
        for instr_idx, (instr, line_num) in enumerate(block.instrlist):
            new_instr = copy.deepcopy(instr)
            
            if isinstance(new_instr, AssignmentCommand):
                # Replace uses in right-hand side
                new_instr.rexpr = self._replace_uses(new_instr.rexpr)
                
                # Create new version for the defined variable
                var_name = new_instr.lvar.varname
                new_version = self._new_variable_version(var_name)
                new_instr.lvar = Var(f"{var_name}_{new_version}")
                
                # Update repeat counter version if applicable
                if self.repeat_handler.is_repeat_counter(var_name):
                    self.repeat_handler.increment_version(var_name)
                
                # Update the block's instruction
                block.instrlist[instr_idx] = (new_instr, line_num)
            elif hasattr(new_instr, 'expr'):
                # Replace uses in expressions
                new_instr.expr = self._replace_uses(new_instr.expr)
                block.instrlist[instr_idx] = (new_instr, line_num)
            elif hasattr(new_instr, 'cond'):
                # Replace uses in conditions
                new_instr.cond = self._replace_uses(new_instr.cond)
                block.instrlist[instr_idx] = (new_instr, line_num)
        
        # Process successors and their phi functions BEFORE recursively processing children
        self._update_phi_functions_in_successors(block)
        
        # Save the current state of the stack for all variables
        stack_snapshot = {}
        for var_name, versions in self.stack.items():
            stack_snapshot[var_name] = versions.copy()
        
        # Recursively process dominator children
        for child in dom_children[block]:
            self._rename_variables_recursive(child, dom_children)
            
            # After processing each child, restore the stack to the state before processing it
            for var_name, versions in stack_snapshot.items():
                self.stack[var_name] = versions.copy()

    def _update_phi_functions_in_successors(self, block):
        """
        Update phi functions in all successors, handling paths more thoroughly
        """
        # Identify all variables defined in this block
        block_definitions = {}
        for instr, _ in block.instrlist:
            if isinstance(instr, AssignmentCommand):
                var_name = instr.lvar.varname
                block_definitions[var_name.split('_')[0]] = var_name
        
        # For each successor, update phi functions directly
        for succ in self.cfg.successors(block):
            edge_label = self.cfg.get_edge_label(block, succ) or "default"
            if hasattr(succ, 'phi_functions'):
                for phi in succ.phi_functions:
                    phi_var_name = phi.target_var.varname.split('_')[0]
                    # Use the latest version of this variable from this block or its path
                    if phi_var_name in block_definitions:
                        source_var = Var(block_definitions[phi_var_name])
                        # Update or add the source
                        updated = False
                        for i, (src, edge) in enumerate(phi.source_vars):
                            if edge == edge_label:
                                phi.source_vars[i] = (source_var, edge_label)
                                updated = True
                                break
                        if not updated:
                            phi.source_vars.append((source_var, edge_label))
        
        # Now trace paths through blocks that don't define these variables
        visited = set([block])
        self._trace_paths_to_phi_functions(block, block_definitions, visited)

    def _trace_paths_to_phi_functions(self, block, var_definitions, visited):
        """
        Trace paths from this block to find phi functions that need updating
        """
        for succ in self.cfg.successors(block):
            if succ in visited:
                continue
            
            visited.add(succ)
            edge_label = self.cfg.get_edge_label(block, succ) or "default"
            
            # Check if this block defines any variables we're tracking
            block_redefines = False
            for instr, _ in succ.instrlist:
                if isinstance(instr, AssignmentCommand):
                    var_name = instr.lvar.varname.split('_')[0]
                    if var_name in var_definitions:
                        block_redefines = True
                        # Update our definitions map with the new version
                        var_definitions[var_name] = instr.lvar.varname
            
            # Update phi functions in this successor
            if hasattr(succ, 'phi_functions'):
                for phi in succ.phi_functions:
                    phi_var_name = phi.target_var.varname.split('_')[0]
                    if phi_var_name in var_definitions:
                        source_var = Var(var_definitions[phi_var_name])
                        # Update or add the source
                        updated = False
                        for i, (src, edge) in enumerate(phi.source_vars):
                            if edge == edge_label:
                                phi.source_vars[i] = (source_var, edge_label)
                                updated = True
                                break
                        if not updated:
                            phi.source_vars.append((source_var, edge_label))
            
            # Continue tracing if this block doesn't redefine our variables
            if not block_redefines:
                self._trace_paths_to_phi_functions(succ, var_definitions.copy(), visited.copy())

    def _new_variable_version(self, var_name):
        """
        Create a new version of a variable and push it onto the stack
        """
        # Initialize the stack entry if it doesn't exist yet
        if var_name not in self.stack:
            self.stack[var_name] = []
        
        self.counter[var_name] += 1
        new_version = self.counter[var_name]
        self.stack[var_name].append(new_version)
        return new_version
    
    def _replace_uses(self, expr):
        """
        Replace variable uses with their current version
        """
        if isinstance(expr, Var):
            var_name = expr.varname
            # Skip SSA-renamed variables (already have version)
            if '_' in var_name and var_name.split('_')[-1].isdigit():
                return expr
                
            # Use the current version from stack
            if self.stack[var_name]:
                current_version = self.stack[var_name][-1]
                return Var(f"{var_name}_{current_version}")
            else:
                # Special handling for repeat counters
                if self.repeat_handler.is_repeat_counter(var_name):
                    current_version = self.repeat_handler.get_current_version(var_name)
                    if current_version > 0:
                        return Var(f"{var_name}_{current_version}")
                    return Var(f"{var_name}_1")  # Default to version 1 for repeat counters
                else:
                    return Var(f"{var_name}_0")
        elif hasattr(expr, 'lexpr') and hasattr(expr, 'rexpr'):
            # Binary operations
            expr.lexpr = self._replace_uses(expr.lexpr)
            expr.rexpr = self._replace_uses(expr.rexpr)
            return expr
        elif hasattr(expr, 'expr'):
            # Unary operations
            expr.expr = self._replace_uses(expr.expr)
            return expr
        else:
            # Constants or other expressions
            return expr

    def _fix_phi_functions(self):
        """Ensure all phi functions have proper sources from all predecessors"""
        for block in self.cfg.nodes():
            if not hasattr(block, 'phi_functions'):
                continue
                
            preds = list(self.cfg.predecessors(block))
            
            for phi in block.phi_functions:
                base_var_name = phi.target_var.varname.split('_')[0]
                
                # Special handling for repeat counter variables
                if self.repeat_handler.is_repeat_counter(base_var_name):
                    phi.source_vars = []
                    
                    for pred in preds:
                        edge_label = self.cfg.get_edge_label(pred, block) or "default"
                        version = self._find_latest_version(pred, base_var_name)
                        source_var = Var(f"{base_var_name}_{version}")
                        phi.source_vars.append((source_var, edge_label))
                    continue
                
                pred_edge_map = {}
                for source, edge in phi.source_vars:
                    for pred in preds:
                        edge_label = self.cfg.get_edge_label(pred, block) or "default"
                        if edge == edge_label:
                            pred_edge_map[pred] = (source, edge)
                
                for pred in preds:
                    edge_label = self.cfg.get_edge_label(pred, block) or "default"
                    if pred not in pred_edge_map:
                        version = self._find_latest_version(pred, base_var_name)
                        source_var = Var(f"{base_var_name}_{version}")
                        phi.source_vars.append((source_var, edge_label))
                        
                phi.source_vars.sort(key=lambda x: x[1])
    
    def _find_latest_version(self, block, var_name):
        """
        Find the latest version of a variable in a block by looking at its instructions
        """
        version = 0
        
        for instr, _ in block.instrlist:
            if isinstance(instr, AssignmentCommand):
                lvar_name = instr.lvar.varname
                lvar_parts = lvar_name.split('_')
                
                if lvar_parts[0] == var_name and len(lvar_parts) > 1:
                    try:
                        current_version = int(lvar_parts[-1])
                        version = max(version, current_version)
                    except ValueError:
                        pass
        
        if hasattr(block, 'phi_functions'):
            for phi in block.phi_functions:
                phi_var_name = phi.target_var.varname
                phi_parts = phi_var_name.split('_')
                
                if phi_parts[0] == var_name and len(phi_parts) > 1:
                    try:
                        current_version = int(phi_parts[-1])
                        version = max(version, current_version)
                    except ValueError:
                        pass
        
        if version == 0 and var_name.startswith('__rep_counter_'):
            return 1
            
        return version
        
    def optimize_block_structure(self):
        """
        Optimize block structure by combining consecutive blocks when possible
        """
        continue_optimizing = True
        
        while continue_optimizing:
            continue_optimizing = False
            
            for block in list(self.cfg.nodes()):
                succs = list(self.cfg.successors(block))
                if len(succs) == 1:
                    succ = succs[0]
                    preds = list(self.cfg.predecessors(succ))
                    
                    if len(preds) == 1 and block != succ:
                        if succ in self.loop_headers:
                            continue
                            
                        block.instrlist.extend(succ.instrlist)
                        
                        for s in list(self.cfg.successors(succ)):
                            edge_label = self.cfg.get_edge_label(succ, s)
                            self.cfg.add_edge(block, s, label=edge_label)
                        
                        self.cfg.nxgraph.remove_node(succ)
                        continue_optimizing = True
                        break
    
    def transform(self):
        """
        Main method to transform CFG to SSA form
        """
        self.dominators = self.compute_dominators()
        self.dom_tree = self.build_dominance_tree(self.dominators)
        self.dominance_frontiers = self.compute_dominance_frontiers(self.dom_tree)
        
        self.identify_loops()
        
        self.repeat_handler.analyze_repeat_counters()
        self.repeat_handler.initialize_repeat_counters()
        
        self.analyze_variables()
        
        self.place_phi_functions()
        
        self.rename_variables()
        
        self._fix_phi_functions()
      
        self.optimize_block_structure()
        
        self._convert_phi_functions()
        
        return self.cfg
    
    def _convert_phi_functions(self):
        """
        Convert phi functions to assignments and handle control flow
        """
        for block in self.cfg.nodes():
            if not hasattr(block, 'phi_functions'):
                continue
            
            phi_instructions = []
            for phi in block.phi_functions:
                # Fix edge labels and remove duplicates
                edge_to_var = {}
                for src, edge in phi.source_vars:
                    edge_to_var[edge] = src
                
                # Rebuild source_vars without duplicates
                phi.source_vars = [(src, edge) for edge, src in edge_to_var.items()]
                
                # Check if we need to add any default sources
                if len(phi.source_vars) == 0:
                    var_name = phi.target_var.varname.split('_')[0]
                    phi.source_vars.append((Var(f"{var_name}_0"), "default"))
                
                phi_instr = AssignmentCommand(phi.target_var, phi)
                phi_instructions.append((phi_instr, -1))
            
            block.instrlist = phi_instructions + block.instrlist
            del block.phi_functions

def convert_to_ssa(cfg):
    """
    Convert a CFG to SSA form
    """
    print("Converting CFG to SSA form...")
    transformer = SSATransformer(cfg)
    result = transformer.transform()
    print("SSA transformation completed.")
    
    print("\nSSA-transformed code:")
    for block in cfg.nodes():
        print(f"\nBlock {block.name}:")
        for instr, _ in block.instrlist:
            if isinstance(instr, AssignmentCommand) and isinstance(instr.rexpr, PhiFunction):
                source_vars = []
                for src_var, edge_label in instr.rexpr.source_vars:
                    if edge_label == "default":
                        source_vars.append(f"{src_var.varname}")
                    else:
                        source_vars.append(f"{src_var.varname} ({edge_label})")
                
                if source_vars:
                    phi_args = ", ".join(source_vars)
                    print(f"  {instr.lvar.varname} = Φ({phi_args})")
                else:
                    print(f"  {instr.lvar.varname} = Φ()")
            elif isinstance(instr, AssignmentCommand):
                print(f"  {instr.lvar.varname} = {instr.rexpr}")
            elif hasattr(instr, 'cond'):
                print(f"  if ({instr.cond})")
            else:
                print(f"  {instr}")
    
    return result
