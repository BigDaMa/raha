/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.lab.hc;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Operation;
import qa.qcri.nadeef.core.pipeline.ExecutionContext;
import qa.qcri.nadeef.core.pipeline.FixDecisionMaker;

import java.util.*;

/**
 * Holistic Cleaning algorithm based on paper
 * Holistic Data Cleaning: Putting Violations Into Context
 * (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6544847&tag=1)
 */
public class HolisticCleaning extends FixDecisionMaker {
    private HashMap<Cell, HashSet<Fix>> graphMap = Maps.newHashMap();

    public HolisticCleaning(ExecutionContext context) {
        super(context);
    }

    /**
     * CountPair is used to generate a Min/Max heap with {@link PriorityQueue}.
     */
    class CountPair<T> implements Comparable<CountPair> {
        T cell;
        int count;

        @Override
        public int compareTo(CountPair o) {
            // turns minHeap to maxHeap
            return -Integer.compare(count, o.count);
        }

        public CountPair(T cell, int count) {
            this.cell = cell;
            this.count = count;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Fix> decide(Collection<Fix> fixes) {
        // generate the hyper-graph
        // vertex is the cell, and edge is the fix which works on this cell
        for (Fix fix : fixes) {
            Cell cell = fix.getLeft();
            addOrCreate(cell, fix);

            if (!fix.isRightConstant()) {
                Cell rightCell = fix.getRight();
                addOrCreate(rightCell, fix);
            }
        }

        // initialize the heap by count of edges
        // Here we use a greedy approach to find the MVC, which is
        // to get the vertexes in the order of hyper-edge counts.
        PriorityQueue<CountPair<Cell>> maxHeap = new PriorityQueue<>();
        HashMap<Cell, CountPair<Cell>> heapIndex = Maps.newHashMap();

        for (Cell cell: graphMap.keySet()) {
            CountPair<Cell> pair = new CountPair<>(cell, graphMap.get(cell).size());
            maxHeap.add(pair);
            heapIndex.put(cell, pair);
        }

        ArrayList<Fix> result = Lists.newArrayList();

        // we try the node with maximum connected fixes first
        // until the heap is empty.
        while (!maxHeap.isEmpty()) {
            HashSet<Fix> repairContext = Sets.newHashSet();
            HashSet<Integer> vids = Sets.newHashSet();

            CountPair<Cell> topCell = maxHeap.poll();
            // generate the frontier using BFS
            HashSet<Cell> frontier = generateFrontier(topCell.cell);

            // generate repairContext and record all the hyper edges
            for (Cell cell : frontier) {
                HashSet<Fix> fixSet = graphMap.get(cell);
                repairContext.addAll(fixSet);
                for (Fix fix : fixSet)
                    vids.add(fix.getVid());
            }

            List<Fix> possibleFixes = determine(repairContext, frontier, topCell.cell);
            if (possibleFixes.size() > 0) {
                // remove hyper-edges when there is a solution
                result.addAll(possibleFixes);

                for (Fix fix : fixes) {
                    int vid = fix.getVid();
                    if (vids.contains(vid)) {
                        Cell cell = fix.getLeft();
                        if (heapIndex.containsKey(cell))
                            maxHeap.remove(heapIndex.get(cell));
                        if (!fix.isRightConstant() && heapIndex.containsKey(cell)) {
                            cell = fix.getRight();
                            maxHeap.remove(heapIndex.get(cell));
                        }
                    }
                }
            }
        }

        return result;
    }

    private void addOrCreate(Cell cell, Fix fix) {
        if (graphMap.containsKey(cell)) {
            HashSet<Fix> relatedFix = graphMap.get(cell);
            relatedFix.add(fix);
        } else {
            HashSet<Fix> relatedFix = Sets.newHashSet();
            relatedFix.add(fix);
            graphMap.put(cell, relatedFix);
        }
    }

    private HashSet<Cell> generateFrontier(Cell topCell) {
        HashSet<Cell> result = Sets.newHashSet();
        result.add(topCell);

        Queue<Fix> queue = new LinkedList<>(graphMap.get(topCell));
        while (!queue.isEmpty()) {
            Fix fix = queue.poll();
            Cell leftCell = fix.getLeft();
            if (!result.contains(leftCell)) {
                result.add(leftCell);
                queue.addAll(graphMap.get(leftCell));
            }

            if (!fix.isRightConstant()) {
                Cell rightCell = fix.getRight();
                if (!result.contains(rightCell)) {
                    result.add(rightCell);
                    queue.addAll(graphMap.get(rightCell));
                }
            }
        }
        return result;

    }

    private List<Fix> determine(
        HashSet<Fix> repairContext,
        HashSet<Cell> frontier,
        Cell topCell
    ) {
        boolean hasOnlyEq = true;
        for (Fix fix : repairContext) {
            if (fix.getOperation() != Operation.EQ &&
                fix.getOperation() != Operation.NEQ
            ) {
                hasOnlyEq = false;
                break;
            }
        }

        if (hasOnlyEq)
            // executing vfm
            return new VFMSolver().solve(repairContext);

        // executing QP, combinatorial trail
        GurobiSolver solver = new GurobiSolver();
        CombinationGenerator<Cell> gen = new CombinationGenerator<>(frontier);
        HashSet<Cell> trail = gen.getNext();
        List<Fix> result = null;
        while (trail != null) {
            result = solver.solve(repairContext, trail);
            if (result != null &&
                result.size() > 0 &&
                FixExtensions.isValidFix(repairContext, result))
                break;
            trail = gen.getNext();
        }

        // when everything fail, we mark it as not resolvable.
        // How?
        // mark topCell as the same value as before, this leads to the same assignment
        // in the next cleaning iteration, which eventually leads to a "fresh value".
        // TODO: please prove its correctness.
        if (result == null) {
            result = Lists.newArrayList();
            Fix fix =
                new Fix.Builder()
                .left(topCell)
                .right(topCell)
                .op(Operation.EQ).build();
            result.add(fix);
        }

        return result;
    }
}