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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.sat4j.core.VecInt;
import org.sat4j.maxsat.SolverFactory;
import org.sat4j.maxsat.WeightedMaxSatDecorator;
import org.sat4j.pb.IPBSolver;
import org.sat4j.specs.IProblem;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.tools.Logger;

import java.io.PrintWriter;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class SatSolver extends FixDecisionMaker {
    private HashMap<SatVariable, Integer> variableIndexMap;
    private List<SatVariable> satVariableList;

    private class SatVariableFactory {
        public int createSatVariable(Cell cell, Object value) {
            SatVariable variable = new SatVariable();
            variable.cell = cell;
            variable.assignedValue = value;

            int index;
            if (variableIndexMap.containsKey(variable)) {
                index = variableIndexMap.get(variable);
            } else {
                satVariableList.add(variable);
                index = satVariableList.size() - 1;
                variableIndexMap.put(variable, index);
            }
            return index;
        }
    }

    private class SatVariable {
        Cell cell;
        Object assignedValue;

        private SatVariable() {}

        @Override
        public int hashCode() {
            return cell.hashCode() * assignedValue.hashCode();
        }

        @Override
        public boolean equals(Object object) {
            if (object == null || !(object instanceof SatVariable))
                return false;
            SatVariable variable = (SatVariable)object;
            return
                cell.equals(variable.cell) &&
                assignedValue.equals(variable.assignedValue);
        }
    }

    public SatSolver(ExecutionContext context) {
        super(context);
        variableIndexMap = Maps.newHashMap();
        satVariableList = Lists.newArrayList();

        // insert a dummy entry to prevent a value of 0, which
        // to avoid invalid index in Sat solver.
        satVariableList.add(null);
    }

    @Override
    public Collection<Fix> decide(Collection<Fix> fixes) {
        // cluster of Fixes based on sharing cells.
        List<HashSet<Fix>> clusters = Lists.newArrayList();
        // map between cell and related fixes.
        HashMap<Cell, HashSet<Fix>> clusterMap = Maps.newHashMap();

        // cluster fixes by sharing cells.
        for (Fix fix : fixes) {
            Cell leftCell = fix.getLeft();
            HashSet<Fix> leftFixCluster = null;
            if (clusterMap.containsKey(leftCell)) {
                leftFixCluster = clusterMap.get(leftCell);
            }

            Cell rightCell = fix.isRightConstant() ? null : fix.getRight();
            HashSet<Fix> rightFixCluster = null;
            if (rightCell != null && clusterMap.containsKey(rightCell)) {
                rightFixCluster = clusterMap.get(rightCell);
            }

            if (rightCell == null) {
                // right hand is a constant
                if (leftFixCluster == null) {
                    HashSet<Fix> newCluster = Sets.newHashSet();
                    newCluster.add(fix);
                    clusterMap.put(leftCell, newCluster);
                    clusters.add(newCluster);
                } else {
                    HashSet<Fix> cluster = clusterMap.get(leftCell);
                    cluster.add(fix);
                }
            } else if (leftFixCluster == null && rightFixCluster == null) {
                // left cell is new and right cell is new
                HashSet<Fix> newCluster = Sets.newHashSet();
                newCluster.add(fix);
                clusterMap.put(leftCell, newCluster);
                clusterMap.put(rightCell, newCluster);
                clusters.add(newCluster);
            } else if (leftFixCluster == null) {
                // left cell is new but right cell is not new
                rightFixCluster.add(fix);
                clusterMap.put(leftCell, rightFixCluster);
            } else if (rightFixCluster == null) {
                // left cell is not new but right cell is new
                leftFixCluster.add(fix);
                clusterMap.put(rightCell, leftFixCluster);
            }
        }

        // deal with each cluster
        List<VecInt> softClauses = Lists.newArrayList();
        List<VecInt> hardClauses = Lists.newArrayList();

        for (HashSet<Fix> cluster : clusters) {
            HashMap<Cell, HashSet<Integer>> litmap = Maps.newHashMap();
            HashSet<Cell> cells = Sets.newHashSet();

            for (Fix fix : cluster) {
                // generate inclusive clauses
                // for assignment like
                //      t1.a (v1) = t2.a (v2) (or constant c)
                // generate clauses like
                // (t1.a = v1 v t1.a = v2) ^ (~t1.a = v1 v ~t1.a = v2)
                Cell leftCell = fix.getLeft();
                int lit1 =
                    new SatVariableFactory().createSatVariable(
                        leftCell,
                        leftCell.getValue()
                    );
                Object value =
                    fix.isRightConstant() ? fix.getRightValue() : fix.getRight().getValue();
                int lit2 =
                    new SatVariableFactory()
                        .createSatVariable(leftCell, value);
                softClauses.add(new VecInt(new int[]{lit1, lit2}));
                // softClauses.add(new VecInt(new int[]{-lit1, lit2}));

                HashSet<Integer> lits;
                if (!litmap.containsKey(leftCell)) {
                    lits = Sets.newHashSet();
                    litmap.put(leftCell, lits);
                } else {
                    lits = litmap.get(leftCell);
                }

                lits.add(lit1);
                lits.add(lit2);
                cells.add(leftCell);

                if (!fix.isRightConstant()) {
                    // for non-constant assignment
                    // generate
                    //      (t2.a = v1 v t2.a = v2) ^ (~t2.a = v1 v ~t2.a = v2)
                    Cell rightCell = fix.getRight();
                    int lit3 =
                        new SatVariableFactory()
                            .createSatVariable(
                                rightCell,
                                rightCell.getValue()
                            );
                    int lit4 =
                        new SatVariableFactory()
                            .createSatVariable(
                                rightCell,
                                leftCell.getValue()
                            );
                    softClauses.add(new VecInt(new int[]{lit3, lit4}));
                    // softClauses.add(new VecInt(new int[]{-lit3, lit4}));

                    if (!litmap.containsKey(rightCell)) {
                        lits = Sets.newHashSet();
                        litmap.put(rightCell, lits);
                    } else {
                        lits = litmap.get(rightCell);
                    }

                    lits.add(lit3);
                    lits.add(lit4);
                    cells.add(rightCell);
                }
            }

            // generate exclusive clauses
            // for each sat variable with the same cell
            // we need to generate exclusive clauses like
            // (~t1.a = v1 v ~t1.a = v2) for C(n, 2) pairs
            // to prevent a variable is assigned two values at the same time.
            for (Cell cell : cells) {
                HashSet<Integer> lits = litmap.get(cell);
                Object[] lists = lits.toArray();
                for (int i = 0; i < lists.length; i ++)
                    for (int j = i + 1; j < lists.length; j ++) {
                        int c1 = (Integer)lists[i];
                        int c2 = (Integer)lists[j];
                        hardClauses.add(new VecInt(new int[]{-c1, -c2}));
                    }
            }

            // violation avoidance
            // finally, we need to generate violation avoidance clauses,
            // which are lits for current violated cells.
            HashMap<Integer, List<Integer>> vioGroup = Maps.newHashMap();
            for (Fix fix : cluster) {
                int vid = fix.getVid();
                int lit1 =
                    new SatVariableFactory()
                        .createSatVariable(fix.getLeft(), fix.getLeft().getValue());
                List<Integer> lits;
                if (vioGroup.containsKey(vid)) {
                    lits = vioGroup.get(vid);
                    lits.add(lit1);
                } else {
                    lits = Lists.newArrayList();
                    lits.add(lit1);
                    vioGroup.put(vid, lits);
                }

                if (!fix.isRightConstant()) {
                    int lit2 =
                        new SatVariableFactory()
                            .createSatVariable(
                                fix.getRight(),
                                fix.getRight().getValue()
                            );
                    lits.add(lit2);
                }
            }

            for (List<Integer> group : vioGroup.values()) {
                int[] lits = new int[group.size()];
                int i = 0;
                for (Integer lit : group) {
                    lits[i ++] = -lit;
                }
                softClauses.add(new VecInt(lits));
            }
        }

        // running sat
        List<Fix> newFix = Lists.newArrayList();
        try {
            IPBSolver solver = SolverFactory.newDefault();
            WeightedMaxSatDecorator decorator = new WeightedMaxSatDecorator(solver);
            decorator.newVar(satVariableList.size());
            decorator.setExpectedNumberOfClauses(softClauses.size());

            for (VecInt clause : softClauses) {
                decorator.addClause(clause);
            }

            for (VecInt clause : hardClauses) {
                decorator.addHardClause(clause);
            }

            IProblem problem = decorator;
            boolean issat = problem.isSatisfiable();
            solver.printInfos(new PrintWriter(System.out));

            if (issat) {
                int[] model = problem.model();
                // match the solution
                for (int i = 0; i < model.length; i ++) {
                    if (model[i] > 0) {
                        SatVariable variable = satVariableList.get(model[i]);
                        newFix.add(
                            new Fix.Builder()
                                .left(variable.cell)
                                .right(variable.assignedValue)
                                .build()
                        );
                    }
                }
            }
        } catch (Exception ex) {
            Logger tracer = Logger.getLogger(this.getClass());
            tracer.error("Sat solving failed.", ex);
        }
        return newFix;
    }
}
