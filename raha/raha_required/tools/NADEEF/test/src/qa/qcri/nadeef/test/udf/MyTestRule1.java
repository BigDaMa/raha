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

public class MyTestRule1 extends SingleTupleRule {
    @Override
    public Collection<Violation> detect(Tuple tuple) {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return Lists.newArrayList();
    }

    @Override
    public Collection<Fix> repair(Violation violation) {
        return null;
    }

    @Override
    public void iterator(Collection<Table> blocks, IteratorResultHandler iteratorResultHandler) {
        Table table = blocks.iterator().next();
        for (int i = 0; i < table.size(); i ++) {
            iteratorResultHandler.handle(table.get(i));
        }
    }
}
