"""Deterministic, gold-aware diagnostics for comparing two fixed-K decoders.

The module deliberately has no data or model dependencies.  ``compare`` is
intended for one document and accepts chronological ``(start, end, label)``
tuples.  Gold is used only for evaluation and diagnostics.
"""


def _triples(rows):
    result = [tuple(row) for row in rows]
    if any(len(row) != 3 for row in result):
        raise ValueError("entities must be (start, end, label) triples")
    if result != sorted(result):
        raise ValueError("entities must be chronological")
    if len(set(result)) != len(result):
        raise ValueError("duplicate entities")
    previous_end = 0
    for start, end, label in result:
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or start < previous_end or not label:
            raise ValueError("invalid, overlapping or non-positive span")
        previous_end = end
    return result


def _rank(rows):
    return {span: index + 1 for index, span in enumerate(rows)}


def _common(gold, pred):
    pr = _rank(pred)
    return [
        {
            "gold": span,
            "gold_rank": i,
            "pred_rank": pr[span],
            "displacement": pr[span] - i,
        }
        for i, span in enumerate(gold, 1) if span in pr
    ]


def _gap_audit(gold, pred, common):
    """Return gap accounting and assert the displacement invariant."""
    gold_rank = _rank(gold)
    pred_rank = _rank(pred)
    ordered = [{"gold_rank": 0, "pred_rank": 0, "displacement": 0}]
    ordered.extend(common)
    ordered.append({"gold_rank": len(gold) + 1, "pred_rank": len(pred) + 1,
                    "displacement": 0})
    gaps = []
    cumulative = 0
    for left, right in zip(ordered, ordered[1:]):
        left_g, right_g = left["gold_rank"], right["gold_rank"]
        left_p, right_p = left["pred_rank"], right["pred_rank"]
        p = sum(span not in gold for span in pred[left_p:right_p - 1])
        g = sum(span not in pred for span in gold[left_g:right_g - 1])
        imbalance = p - g
        cumulative += imbalance
        expected = right["pred_rank"] - right["gold_rank"]
        if cumulative != expected:
            raise AssertionError("cumulative gap imbalance does not equal displacement")
        gaps.append({"left_gold_rank": left_g, "right_gold_rank": right_g,
                     "pred_noncommon": p, "gold_noncommon": g,
                     "imbalance": imbalance, "cumulative": cumulative})
    return gaps


def _episodes(common, gold):
    """Find maximal nonzero runs in the common-anchor displacement series."""
    runs = []
    complex_runs = []
    anchors = [{"gold_rank": 0, "pred_rank": 0, "displacement": 0}] + list(common)
    anchors.append({"gold_rank": len(gold) + 1, "pred_rank": len(gold) + 1, "displacement": 0})
    i = 1
    while i < len(anchors) - 1:
        if anchors[i]["displacement"] == 0:
            i += 1
            continue
        j = i
        while j < len(anchors) - 1 and anchors[j]["displacement"] != 0:
            j += 1
        block = anchors[i:j]
        displacements = [x["displacement"] for x in block]
        payload = {"gold_ranks": [x["gold_rank"] for x in block],
                   "entities": [x["gold"] for x in block],
                   "displacements": displacements, "length": len(block)}
        if all(x == 1 for x in displacements) or all(x == -1 for x in displacements):
            payload["displacement"] = displacements[0]
            payload["absolute_displacement_one"] = len(block)
            runs.append(payload)
        else:
            complex_runs.append(payload)
        i = j
    return runs, complex_runs


def compare(gold, map_pred, mbr_pred):
    """Compare MAP and MBR predictions for one document.

    All returned values are JSON-safe.  Episode entity sets are fixed from
    MAP, so dropping an entity can never be counted as a repair.
    """
    gold, mp, bp = _triples(gold), _triples(map_pred), _triples(mbr_pred)
    common_map, common_mbr = _common(gold, mp), _common(gold, bp)
    if not (len(gold) == len(mp) == len(bp)):
        raise ValueError("MAP and MBR must have the same fixed K as gold")
    gaps = _gap_audit(gold, mp, common_map)
    mbr_gaps = _gap_audit(gold, bp, common_mbr)
    episodes, complex_runs = _episodes(common_map, gold)
    mbr_episodes, mbr_complex_runs = _episodes(common_mbr, gold)
    mr, br = _rank(mp), _rank(bp)
    gr = _rank(gold)
    episode_results = []
    repaired = retained = lost = 0
    for episode in episodes:
        refs = episode["entities"]
        states = []
        for span in refs:
            if span not in br:
                state = "LOST"; lost += 1
            elif br[span] == gr[span]:
                state = "REPAIRED"; repaired += 1
            else:
                state = "RETAINED_DISPLACED"; retained += 1
            states.append({"entity": span, "state": state})
        episode_results.append({**episode, "entities": states,
                                "complete_repair": all(x["state"] == "REPAIRED" for x in states),
                                "repaired": sum(x["state"] == "REPAIRED" for x in states),
                                "retained_displaced": sum(x["state"] == "RETAINED_DISPLACED" for x in states),
                                "lost": sum(x["state"] == "LOST" for x in states)})
    map_exact = set(gold) & set(mp)
    mbr_exact = set(gold) & set(bp)
    map_correct = {x for x in map_exact if mr[x] == gr[x]}
    mbr_correct = {x for x in mbr_exact if br[x] == gr[x]}
    map_displaced = map_exact - map_correct
    mbr_displaced = mbr_exact - mbr_correct
    transitions = {
        "map_correct_to_mbr_correct": len(map_correct & mbr_correct),
        "map_correct_to_mbr_wrong": len(map_correct & mbr_displaced),
        "map_correct_to_mbr_lost": len(map_correct - mbr_exact),
        "map_wrong_to_mbr_correct": len(map_displaced & mbr_correct),
        "map_wrong_to_mbr_wrong": len(map_displaced & mbr_displaced),
        "map_wrong_to_mbr_lost": len(map_displaced - mbr_exact),
        "map_absent_to_mbr_correct": len(mbr_correct - map_exact),
        "map_absent_to_mbr_wrong": len(mbr_displaced - map_exact),
        "map_absent_to_mbr_absent": len(set(gold) - map_exact - mbr_exact),
    }
    if sum(transitions.values()) != len(gold):
        raise AssertionError("entity transitions must partition all gold")
    if repaired + retained + lost != sum(x['length'] for x in episodes):
        raise AssertionError("episode outcomes must partition fixed MAP references")
    newly_wrong = map_correct & mbr_displaced
    new_displaced = (set(gold) - map_displaced) & mbr_displaced
    new_episode_signatures = {(tuple(x["entities"]), x["displacement"]) for x in episodes}
    new_episodes = [x for x in mbr_episodes if (tuple(x["entities"]), x["displacement"]) not in new_episode_signatures]
    def model_stats(rows, simple):
        exact = set(gold) & set(rows)
        correct = {x for x in exact if _rank(rows)[x] == gr[x]}
        return {"exact_entities": len(exact), "correct_entities": len(correct),
                "displaced_entities": len(exact - correct),
                "document_has_displacement": bool(exact - correct),
                "absolute_displacement_one": sum(abs(_rank(rows)[x] - gr[x]) == 1 for x in exact),
                "episodes": len(simple), "simple_episode_entities": sum(x['length'] for x in simple),
                "episode_lengths": [x["length"] for x in simple]}
    return {
        "gold_count": len(gold), "map_count": len(mp), "mbr_count": len(bp),
        "common_entities": common_map, "mbr_common_entities": common_mbr,
        "gaps": gaps, "mbr_gaps": mbr_gaps, "simple_episodes": episode_results,
        "complex_displacement_runs": complex_runs,
        "mbr_simple_episodes": mbr_episodes, "mbr_complex_displacement_runs": mbr_complex_runs,
        "episode_entity_counts": {"REPAIRED": repaired, "RETAINED_DISPLACED": retained, "LOST": lost},
        "complete_repairs": sum(x["complete_repair"] for x in episode_results),
        "documents_affected": bool(episodes or mbr_episodes),
        "new_displacements": {"count": len(new_displaced), "entities": sorted(new_displaced)},
        "newly_wrong_from_map_correct": {"count": len(newly_wrong), "entities": sorted(newly_wrong)},
        "new_episodes": new_episodes,
        "new_episode_count": len(new_episodes),
        "new_episode_definition": "MBR simple episode has a different exact reference entity set or displacement sign from every MAP simple episode; changed/shortened episodes may count, so this is not by itself a regression measure",
        "model_stats": {"map": model_stats(mp, episodes), "mbr": model_stats(bp, mbr_episodes)},
        "transitions": transitions,
    }
