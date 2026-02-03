import networkx as nx
from classes.LLMUser import LLMUser
from Uncertainpy.src.uncertainpy.gradual import Argument, BAG, semantics, algorithms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
from pathlib import Path
from typing import Dict, List, Tuple
import json
from configuration.hyperparameters import PRUNE_ISOLATED_ARGUMENTS

class ArgumentationGraph:

    def __init__(self, active_hypotheses: bool = False, debug: bool = True):
        self.G = nx.DiGraph()
        self.bag = BAG()
        self.node_text_map: Dict[str, str] = {}  # node_id -> text
        self.node_counter = 0
        self.active_hypotheses = active_hypotheses
        self.debug = debug
        self.last_added_arguments: List[str] = []
        self.last_added_relations: List[Tuple[str, str, str]] = []

    # -----------------------------
    # Add argument safely
    # -----------------------------
    def add_argument(self, text: str, node_type: str = "argument", initial_strength: float = 0.5, node_id: str = None) -> str:
        # Check duplicate
        for nid, t in self.node_text_map.items():
            if t.strip() == text.strip():
                if self.debug:
                    print(f"⚠️ Duplicate argument ignored: {text}")
                return nid

        if node_id is None:
            node_id = f"{node_type[0].upper()}{self.node_counter}"
            self.node_counter += 1
        else:
            # Ensure counter does not collide with given node_id
            try:
                nid_number = int(''.join(filter(str.isdigit, node_id)))
                self.node_counter = max(self.node_counter, nid_number + 1)
            except:
                pass

        self.G.add_node(node_id, type=node_type, text=text, strength=initial_strength)
        self.bag.arguments[node_id] = Argument(node_id, initial_weight=initial_strength)
        self.node_text_map[node_id] = text
        self.last_added_arguments.append(node_id)
        return node_id


    # -----------------------------
    # Add relation safely
    # -----------------------------
    def add_relation(self, src: str, tgt: str, relation: str):
        if self.G.has_edge(src, tgt) and self.G.edges[src, tgt].get("relation") == relation:
            if self.debug:
                print(f"⚠️ Duplicate relation ignored: {src} -> {tgt} : {relation}")
            return
        self.G.add_edge(src, tgt, relation=relation)
        if relation == "support":
            self.bag.add_support(self.bag.arguments[src], self.bag.arguments[tgt])
        elif relation == "attack":
            self.bag.add_attack(self.bag.arguments[src], self.bag.arguments[tgt])
        self.last_added_relations.append((src, tgt, relation))

    # -----------------------------
    # Compute strengths
    # -----------------------------
    def compute_strengths(self) -> Dict[str, float]:
        model = semantics.QuadraticEnergyModel()
        model.BAG = self.bag
        model.approximator = algorithms.RK4(model)
        model.solve(delta=1e-2, epsilon=1e-4)
        strengths: Dict[str, float] = {}
        for arg in self.bag.arguments.values():
            strengths[arg.name] = float(getattr(arg, "strength", arg.get_initial_weight()))
        nx.set_node_attributes(self.G, strengths, "strength")
        return strengths

    # -----------------------------
    # Extend graph from LLM output
    # -----------------------------
    def extend_from_text(self, text: str, llm_user: LLMUser, hypotheses: List[str] = None, max_arguments: int = 3):
        if hypotheses is None:
            hypotheses = []

        self.last_added_arguments = []
        self.last_added_relations = []

        previous_arguments = [self.node_text_map[nid] for nid in self.G.nodes if self.G.nodes[nid]["type"] == "argument"]

        llm_output = llm_user.extract_unified_arguments(
            text=text,
            hypotheses=hypotheses,
            previous_arguments=previous_arguments,
            max_new=max_arguments
        )

        for item in llm_output:
            if not isinstance(item, dict):
                if self.debug:
                    print("⚠️ Skipping non-dict LLM output:", item)
                continue
            if not all(k in item for k in ["text", "relation", "target"]):
                if self.debug:
                    print(f"⚠️ Skipping malformed LLM item (missing keys): {item}")
                continue

            arg_text = str(item["text"]).strip()
            relation = str(item["relation"]).strip()
            target_text = str(item["target"]).strip()
            if not arg_text or not target_text:
                continue

            new_id = self.add_argument(arg_text, "argument")

            # Find or add target
            target_id = None
            for nid, txt in self.node_text_map.items():
                if txt.strip() == target_text:
                    target_id = nid
                    break
            if target_id is None and target_text in hypotheses:
                target_id = self.add_argument(target_text, "hypothesis")

            if target_id:
                self.add_relation(new_id, target_id, relation)
            elif self.debug:
                print(f"⚠️ Relation skipped: target not found → '{target_text}'")

        strengths = self.compute_strengths()
        self.prune_isolated_arguments()

        if self.debug:
            self.print_graph(header="Graph Updated (extend_from_text)")

        return {"strengths": strengths}

    # -----------------------------
    # Utility methods
    # -----------------------------
    def get_text_from_id(self, node_id: str) -> str:
        return self.node_text_map.get(node_id, "")

    def print_graph(self, header: str = "Argumentation Graph State") -> str:
        lines = ["="*60, header, "-"*60, "Nodes (id | type | strength | text):"]
        for nid, data in self.G.nodes(data=True):
            lines.append(f"{nid:4s} | {data.get('type','?'):10s} | {data.get('strength',0):.3f} | {data.get('text','')[:200]}")
        lines.append("\nRelations (src -> tgt : relation):")
        for src, tgt, edata in self.G.edges(data=True):
            lines.append(f"{src} -> {tgt} : {edata.get('relation', '?')}")
        lines.append("="*60+"\n")
        txt = "\n".join(lines)
        print(txt)
        return txt

    # -----------------------------
    # Save & load graph
    # -----------------------------
    def save_graph(self, dataset_name: str, sample_idx: int, save_dir: Path = None):
        graph_dir = (save_dir / dataset_name) if save_dir else (Path("graphs") / dataset_name)
        graph_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = graph_dir / f"sample_{sample_idx}.graphml"
        png_path = graph_dir / f"sample_{sample_idx}.png"
        txt_path = graph_dir / f"sample_{sample_idx}.txt"

        # Save GraphML
        nx.write_graphml(self.G, graphml_path)

        # Draw
        strengths = nx.get_node_attributes(self.G, "strength")
        node_colors = [strengths.get(n, 0.5) for n in self.G.nodes]
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, cmap="coolwarm", node_size=1200, edgecolors="black")
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        attack_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "attack"]
        support_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("relation") == "support"]

        nx.draw_networkx_edges(self.G, pos, edgelist=attack_edges, edge_color="red", arrows=True, arrowsize=20, width=2.0)
        nx.draw_networkx_edges(self.G, pos, edgelist=support_edges, edge_color="green", style="dashed", arrows=True, arrowsize=20, width=1.8)

        # Legend
        legend_text = "Legend (ID | Strength | Text):\n"
        for nid, data in self.G.nodes(data=True):
            legend_text += f"{nid} | {data.get('strength',0):.3f} | {data.get('text','')[:50].replace('\n',' ')}\n"
        plt.gcf().text(0.75, 0.5, legend_text, fontsize=8, va='center', ha='left', wrap=True)
        plt.title(f"{dataset_name} — Sample {sample_idx}")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Save TXT
        lines = ["="*60, f"Graph State: {dataset_name} Sample {sample_idx}", "-"*60]
        lines.append("Nodes (id | type | strength | text):")
        for nid, data in self.G.nodes(data=True):
            lines.append(f"{nid:4s} | {data.get('type','?'):10s} | {data.get('strength',0):.3f} | {data.get('text','')[:200]}")
        lines.append("\nRelations (src -> tgt : relation):")
        for src, tgt, edata in self.G.edges(data=True):
            lines.append(f"{src} -> {tgt} : {edata.get('relation', '?')}")
        lines.append("\nLegend (ID | Strength | Text):")
        for nid, data in self.G.nodes(data=True):
            lines.append(f"{nid:4s} | {data.get('strength',0):.3f} | {data.get('text','')[:50].replace('\n',' ')}")
        lines.append("="*60+"\n")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if self.debug:
            print(f"💾 Saved graph → {graphml_path.name}, {png_path.name}, {txt_path.name}")

    def load_graph(self, graph_path: Path):
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        self.G = nx.read_graphml(graph_path)
        # Restore node_text_map from node attribute 'text'
        self.node_text_map = {nid: self.G.nodes[nid].get("text", "") for nid in self.G.nodes}
        # Restore node_counter to avoid duplicate IDs
        self.node_counter = len(self.G.nodes)
        # Rebuild BAG
        self.bag = BAG()
        for nid in self.G.nodes:
            strength = self.G.nodes[nid].get("strength", 0.5)
            self.bag.arguments[nid] = Argument(nid, initial_weight=strength)
        if self.debug:
            print(f"[INFO] Loaded graph from {graph_path}, nodes={len(self.G.nodes)}")

    # -----------------------------
    # Prune isolated arguments
    # -----------------------------
    def prune_isolated_arguments(self):
        if not PRUNE_ISOLATED_ARGUMENTS:
            return
        isolated = list(nx.isolates(self.G))
        removed = []
        for nid in isolated:
            if self.G.nodes[nid].get("type") == "hypothesis":
                continue
            self.G.remove_node(nid)
            self.bag.arguments.pop(nid, None)
            self.node_text_map.pop(nid, None)
            removed.append(nid)
        if self.debug and removed:
            print(f"🧹 Pruned isolated arguments (excluding hypotheses): {removed}")
