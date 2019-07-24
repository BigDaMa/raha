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

package qa.qcri.nadeef.core.datamodel;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ConcurrentMap;

/**
 * SingleTupleRule rule is an abstract class for rule which has detection based on one tuple.
 */
public abstract class SingleTupleRule extends Rule<Tuple> {
    /**
     * Internal method to initialize a rule.
     * @param name Rule name.
     * @param tableNames Table names.
     */
    public void initialize(String name, List<String> tableNames) {
        super.initialize(name, tableNames);
    }

    /**
     * Default scope operation.
     * @param table input tables.
     * @return filtered table.
     */
    @Override
    public Collection<Table> horizontalScope(Collection<Table> table) {
        return table;
    }

    /**
     * Default scope operation.
     * @param table input table.
     * @return filtered table.
     */
    @Override
    public Collection<Table> verticalScope(Collection<Table> table) {
        return table;
    }

    /**
     * Block operator.
     *
     * Current we don't support blocking given multiple tables (co-group). So when a rule
     * is using more than 1 table the block operator is going to be ignored.
     * @param table a collection of tables.
     * @return a collection of blocked tables.
     */
    @Override
    public Collection<Table> block(Collection<Table> table) {
        return table;
    }

    /**
     * Default iterator operation.
     * @param blocks input table collection.
     */
    @Override
    public void iterator(Collection<Table> blocks, IteratorResultHandler iteratorBlockingQueue) {
        Table table = blocks.iterator().next();
        for (int i = 0; i < table.size(); i ++) {
            iteratorBlockingQueue.handle(table.get(i));
        }
    }

    /**
     * Incremental iterator interface.
     * @param tables blocks.
     * @param newTuples new tuples comes in.
     * @param iteratorResultHandler output stream.
     */
    public final void iterator(
        Collection<Table> tables,
        ConcurrentMap<String, HashSet<Integer>> newTuples,
        IteratorResultHandler iteratorResultHandler
    ) {
        Table table = tables.iterator().next();
        String tableName = table.getSchema().getTableName();

        if (newTuples.containsKey(tableName)) {
            HashSet<Integer> newTuplesIDs = newTuples.get(tableName);
            // iterating all the tuples
            for (int i = 0; i < table.size(); i++) {
                Tuple tuple1 = table.get(i);
                if (newTuplesIDs.contains(tuple1.getTid())) {
                    iteratorResultHandler.handle(tuple1);
                }
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasOwnIterator() {
        boolean result = false;
        try {
            String declareClassName =
                getClass().getMethod(
                    "iterator",
                    new Class[] { Collection.class, IteratorResultHandler.class }
                ).getDeclaringClass().getSimpleName();
            result = !declareClassName.equalsIgnoreCase("SingleTupleRule");
        } catch (Exception ex) {}
        return result;
    }
}