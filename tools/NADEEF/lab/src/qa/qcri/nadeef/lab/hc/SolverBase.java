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

import com.google.common.collect.Sets;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Fix;

import java.util.HashSet;
import java.util.List;

public abstract class SolverBase {

    /**
     * Default solver behavior. It does not restrict the changing cells.
     */
    public List<Fix> solve(HashSet<Fix> repairContext) {
        HashSet<Cell> cells = Sets.newHashSet();
        for (Fix fix : repairContext) {
            cells.add(fix.getLeft());
            if (!fix.isRightConstant())
                cells.add(fix.getRight());
        }
        return solve(repairContext, cells);
    }

    public abstract List<Fix> solve(HashSet<Fix> repairContext, HashSet<Cell> changedCell);
}