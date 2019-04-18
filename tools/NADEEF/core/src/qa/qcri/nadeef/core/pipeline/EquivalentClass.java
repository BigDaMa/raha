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

package qa.qcri.nadeef.core.pipeline;

import com.google.common.collect.*;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;

import java.util.*;

/**
 * EquivalentClass is an implementation of {@link FixDecisionMaker} based on EquivalentClass
 * algorithm.
 *
 */
public class EquivalentClass extends FixDecisionMaker {
    /**
     * Constructor.
     */
    public EquivalentClass(ExecutionContext context) {
        super(context);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Fix> decide(Collection<Fix> fixes) {
        List<HashSet<Cell>> clusters = Lists.newArrayList();
        // a map between a cell and n
        HashMap<Cell, HashSet<Cell>> clusterMap = Maps.newHashMap();
        HashMap<Cell, String> assignMap = Maps.newHashMap();
        // a map between cell and fix, used for getting the original vid.
        HashMap<Cell, Fix> fixMap = Maps.newHashMap();

        // Clustering all the fixes.
        int count = 0;
        for (Fix fix : fixes) {
            Cell leftCell = fix.getLeft();
            fixMap.put(leftCell, fix);

            if (fix.isRightConstant()) {
                // TODO: do a statistic on the assign count.
                assignMap.put(leftCell, fix.getRightValue());
                continue;
            }

            Cell rightCell = fix.getRight();
            fixMap.put(rightCell, fix);
            if (assignMap.containsKey(leftCell)) {
                assignMap.remove(leftCell);
            }

            if (assignMap.containsKey(rightCell)) {
                assignMap.remove(rightCell);
            }

            HashSet<Cell> leftCluster = null;
            HashSet<Cell> rightCluster = null;

            // when the left column is already in a cluster
            if (clusterMap.containsKey(leftCell)) {
                leftCluster = clusterMap.get(leftCell);
                if (!leftCluster.contains(rightCell)) {
                    // union of two cluster of cell sets.
                    if (clusterMap.containsKey(rightCell)) {
                        rightCluster = clusterMap.get(rightCell);
                        for (Cell cell : rightCluster) {
                            leftCluster.add(cell);
                            clusterMap.put(cell, leftCluster);
                        }

                        rightCluster.clear();
                        clusters.remove(rightCluster);
                    } else {
                        clusterMap.put(rightCell, leftCluster);
                        leftCluster.add(rightCell);
                    }
                }
            } else if (clusterMap.containsKey(rightCell)) {
                // when the right column is already in the cluster
                rightCluster = clusterMap.get(rightCell);
                if (!rightCluster.contains(leftCell)) {
                    // union of two cluster of cell sets.
                    if (clusterMap.containsKey(leftCell)) {
                        leftCluster = clusterMap.get(leftCell);
                        for (Cell cell : leftCluster) {
                            rightCluster.add(cell);
                            clusterMap.put(cell, rightCluster);
                        }

                        for (Cell cell : leftCluster) {
                            leftCluster.remove(cell);
                        }

                        clusters.remove(leftCluster);
                    } else {
                        clusterMap.put(leftCell, rightCluster);
                        rightCluster.add(leftCell);
                    }
                }
            } else {
                // both left and right are not in any of the cluster
                // create a new cluster of containing both.
                HashSet<Cell> cluster = Sets.newHashSet();
                cluster.add(leftCell);
                cluster.add(rightCell);
                clusterMap.put(leftCell, cluster);
                clusterMap.put(rightCell, cluster);
                clusters.add(cluster);
            }
        }

        // start to count each cluster and decide the final fix based on
        // percentage.
        List<Fix> result = Lists.newArrayList();
        // for final execution of all the fixes, we use 0 as default as the fix id.
        Fix.Builder fixBuilder = new Fix.Builder();
        count = 0;
        for (HashSet<Cell> cluster : clusters) {
            Multiset<Object> countSet = HashMultiset.create();
            for (Cell cell : cluster) {
                countSet.add(cell.getValue());
            }

            countSet = Multisets.copyHighestCountFirst(countSet);
            Object value = countSet.iterator().next();
            for (Cell cell : cluster) {
                if (cell.getValue().equals(value)) {
                    // skip the correct value.
                    continue;
                }
                Fix originalFix = fixMap.get(cell);
                Fix newFix =
                    fixBuilder.vid(originalFix.getVid())
                        .left(cell)
                        .right(value.toString())
                        .build();
                result.add(newFix);
            }
            count ++;
        }

        // collect the remaining constant assign fix.
        Set<Map.Entry<Cell, String>> entries = assignMap.entrySet();
        for (Map.Entry<Cell, String> entry : entries) {
            Fix newFix = fixBuilder.left(entry.getKey()).right(entry.getValue()).build();
            result.add(newFix);
        }

        setPercentage(1.0f);
        return result;
    }
}
