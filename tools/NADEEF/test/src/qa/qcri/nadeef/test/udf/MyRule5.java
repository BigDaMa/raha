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

package qa.qcri.nadeef.test.udf;

import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.*;

import java.util.Collection;
import java.util.List;

/**
 * Pair table test.
 */
public class MyRule5 extends PairTupleRule {

    /**
     * Detect rule with pair tuple.
     *
     * @param pair input tuple pair.
     * @return Violation set.
     */
    @Override
    public Collection<Violation> detect(TuplePair pair) {
        List<Violation> result = Lists.newArrayList();
        Tuple bankTuple;
        Tuple tranTuple;
        if (pair.getLeft().isFromTable("bank1")) {
            bankTuple = pair.getLeft();
            tranTuple = pair.getRight();
        } else {
            bankTuple = pair.getRight();
            tranTuple = pair.getLeft();
        }

        if (
            bankTuple.get("FN").equals(tranTuple.get("FN")) &&
            bankTuple.get("LN").equals(tranTuple.get("LN")) &&
            bankTuple.get("ST").equals(tranTuple.get("str")) &&
            bankTuple.get("city").equals(tranTuple.get("city")) &&
            !bankTuple.get("tel").equals(tranTuple.get("phn"))
        ) {
            Violation violation = new Violation(getRuleName());
            violation.addCell(bankTuple.getCell("FN"));
            violation.addCell(bankTuple.getCell("LN"));
            violation.addCell(bankTuple.getCell("ST"));
            violation.addCell(bankTuple.getCell("city"));
            violation.addCell(bankTuple.getCell("tel"));
            violation.addCell(tranTuple.getCell("FN"));
            violation.addCell(tranTuple.getCell("LN"));
            violation.addCell(tranTuple.getCell("str"));
            violation.addCell(tranTuple.getCell("city"));
            violation.addCell(tranTuple.getCell("phn"));
            result.add(violation);
        }
        return result;
    }

    /**
     * Repair of this rule.
     *
     * @param violation violation input.
     * @return a candidate fix.
     */
    @Override
    public Collection<Fix> repair(Violation violation) {
        Fix.Builder builder = new Fix.Builder();
        Cell tran = violation.getCell("tran1", "phn");
        Cell bank = violation.getCell("bank1", "tel");
        Fix fix = builder.left(tran).right(bank.getValue().toString()).build();
        return Lists.newArrayList(fix);
    }
}
