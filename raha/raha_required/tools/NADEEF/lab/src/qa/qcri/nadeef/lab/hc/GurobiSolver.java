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
import gurobi.*;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;
import qa.qcri.nadeef.core.datamodel.Operation;
import qa.qcri.nadeef.tools.Logger;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class GurobiSolver extends SolverBase {
    private Logger tracer = Logger.getLogger(GurobiSolver.class);

    public List<Fix> solve(HashSet<Fix> repairContext, HashSet<Cell> changedCell) {
        tracer.fine("Probing: ");
        for (Cell cell : changedCell)
            tracer.fine(cell.getColumn().getFullColumnName() + ":" + cell.getTid());

        GRBEnv env = null;
        GRBModel model = null;
        try {
            env = new GRBEnv();
            env.set(GRB.IntParam.LogToConsole, 0);
            model = new GRBModel(env);

            // encoding QP
            HashMap<Cell, GRBVar> cellIndex = Maps.newHashMap();
            HashMap<GRBVar, Cell> varIndex = Maps.newHashMap();

            int fixIndex = 0;
            // A linear scan to create all the variables and constraints.
            for (Fix fix : repairContext) {
                fixIndex ++;
                Cell cell = fix.getLeft();

                // check left hand of the predicate
                GRBVar var1 = null;
                double v1 = getValue(cell.getValue());
                if (changedCell.contains(cell)) {
                    if (cellIndex.containsKey(cell))
                        var1 = cellIndex.get(cell);
                    else {
                        var1 =
                            model.addVar(
                                Integer.MIN_VALUE,
                                Integer.MAX_VALUE,
                                0.0,
                                GRB.INTEGER,
                                cell.getColumn().getColumnName()
                            );
                        cellIndex.put(cell, var1);
                        varIndex.put(var1, cell);
                    }
                }

                // check right hand of the predicate
                GRBVar var2 = null;
                double v2 = 0.0;
                if (!fix.isRightConstant()) {
                    cell = fix.getRight();
                    v2 = getValue(cell.getValue());
                    if (changedCell.contains(cell)) {
                        if (cellIndex.containsKey(cell))
                            var2 = cellIndex.get(cell);
                        else {
                            var2 =
                                model.addVar(
                                    Integer.MIN_VALUE,
                                    Integer.MAX_VALUE,
                                    0.0,
                                    GRB.INTEGER,
                                    cell.getColumn().getColumnName()
                                );
                            cellIndex.put(cell, var2);
                            varIndex.put(var2, cell);
                        }
                    }
                } else {
                    v2 = getValue(fix.getRightValue());
                }

                // Both cells are in the unmodified state.
                if (var1 == null && var2 == null)
                    continue;

                // update the model terms
                model.update();

                StringBuffer exp = new StringBuffer();
                // constraint construction
                GRBLinExpr constraint = new GRBLinExpr();
                if (var1 != null && var2 != null) {
                    // left and right can both be modified.
                    constraint.addTerm(1.0, var1);
                    constraint.addTerm(-1.0, var2);
                    exp.append(var1.get(GRB.StringAttr.VarName))
                        .append(" - ")
                        .append(var2.get(GRB.StringAttr.VarName));
                } else if (var1 == null) {
                    // left can not be modified, but right can be modified
                    constraint.addConstant(v1);
                    constraint.addTerm(-1.0, var2);
                    exp.append("-")
                        .append(var2.get(GRB.StringAttr.VarName))
                        .append(" + ")
                        .append(v1);
                } else {
                    // left can be modified, but right cannot.
                    constraint.addTerm(1.0, var1);
                    constraint.addConstant(-1.0 * v2);
                    exp.append(var1.get(GRB.StringAttr.VarName))
                        .append(" - ")
                        .append(v2);
                }

                switch (fix.getOperation()) {
                    case EQ:
                        model.addConstr(constraint, GRB.EQUAL, 0.0, "c" + fixIndex);
                        exp.append(" = 0.0");
                        break;
                    // TODO: for a NEQ solution it actually can has both GT and LT,
                    // how to make this fit a QP problem?
                    // a possible solution can be here
                    // http://stackoverflow.com/questions/17257314/integer-programming-unequal-constraint
                    case NEQ:
                    case GT:
                        model.addConstr(constraint, GRB.GREATER_EQUAL, 0.5, "c" + fixIndex);
                        exp.append("> 0.5");
                        break;
                    case GTE:
                        model.addConstr(constraint, GRB.GREATER_EQUAL, 0, "c" + fixIndex);
                        exp.append(">= 0.0");
                        break;
                    case LT:
                        model.addConstr(constraint, GRB.LESS_EQUAL, -0.5, "c" + fixIndex);
                        exp.append("< 0.5");
                        break;
                    case LTE:
                        model.addConstr(constraint, GRB.LESS_EQUAL, 0, "c" + fixIndex);
                        exp.append("<= 0");
                        break;
                }
                tracer.fine(exp.toString());
            }

            // Start optimizing
            model.update();

            // Apply Numerical Distance Minimality
            GRBQuadExpr numMinDistance = new GRBQuadExpr();

            for (Cell cell : cellIndex.keySet()) {
                GRBVar var = cellIndex.get(cell);
                double value = getValue(cell.getValue());
                numMinDistance.addTerm(1.0, var, var);
                numMinDistance.addConstant(value * value);
                numMinDistance.addTerm(-2.0 * value, var);
            }

            // Apply Cardinality Minimality
            /*
            final double constantK = 3.0;
            List<Cell> cells = Lists.newArrayList(cellIndex.keySet());
            for (int i = 0; i < cells.size(); i ++) {
                for (int j = i + 1; j < cells.size(); j ++) {
                    GRBVar var1 = cellIndex.get(cells.get(i));
                    GRBVar var2 = cellIndex.get(cells.get(j));
                    double v1 = getValue(cells.get(i));
                    double v2 = getValue(cells.get(j));

                    numMinDistance.addTerm(constantK, var1, var2);
                    numMinDistance.addTerm(-constantK * v2, var1);
                    numMinDistance.addTerm(-constantK * v1, var2);
                    numMinDistance.addConstant(-constantK * v1 * v2);
                }
            }
            */

            model.setObjective(numMinDistance, GRB.MINIMIZE);


            // Sometimes the model itself has no solution, we need to
            // setup relaxation conditions to achieve maximum constraints.
            // model = model.presolve();
            // model.feasRelax(1, true, true, true);
            model.optimize();

            int status = model.get(GRB.IntAttr.Status);
            if (status == GRB.Status.INFEASIBLE)
                return null;

            // export the output
            List<Fix> fixList = Lists.newArrayList();
            for (GRBVar var : varIndex.keySet()) {
                Cell cell = varIndex.get(var);
                double value = Math.round(var.get(GRB.DoubleAttr.X));
                double oldValue = getValue(cell.getValue());

                // ignore the tiny difference
                if (Math.abs(oldValue - value) > 1e-7) {
                    Fix fix = new Fix.Builder().left(cell).right(value).op(Operation.EQ).build();
                    fixList.add(fix);
                }
            }
            return fixList;
        } catch (Exception ex) {
            tracer.error("Initializing GRB environment failed.", ex);
        }  finally {
            try {
                if (model != null)
                    model.dispose();
                if (env != null)
                    env.dispose();
            } catch (Exception ex) {
                tracer.error("Disposing Gurobi failed.", ex);
            }
        }
        return null;
    }

    private double getValue(Object obj) {
        if (obj instanceof Integer) {
            return (Integer)obj;
        } else if (obj instanceof Double) {
            return (Double)obj;
        } else if (obj instanceof Float) {
            return (Float)obj;
        }
        return Double.parseDouble((String)obj);
    }
}
