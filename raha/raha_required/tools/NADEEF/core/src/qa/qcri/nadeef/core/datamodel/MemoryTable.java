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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import java.util.*;

/**
 * MemoryTable represents a table which resides in memory.
 */
public class MemoryTable extends Table {
    private List<Tuple> tuples;

    private MemoryTable(List<Tuple> tuples) {
        super(tuples.get(0).getSchema());
        this.tuples = tuples;
    }

    public static MemoryTable of(List<Tuple> tuples) {
        Preconditions.checkArgument(tuples != null && tuples.size() > 0);
        return new MemoryTable(tuples);
    }

    //<editor-fold desc="Table override methods">

    /**
     * {@inheritDoc}
     */
    @Override
    public void recycle() {
        tuples.clear();
        tuples = null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return tuples.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tuple get(int i) {
        Preconditions.checkArgument(i >= 0 && i < size());
        return tuples.get(i);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table project(List<Column> columns) {
        Column[] allColumns = schema.getColumns();
        List<Column> toRemove = Lists.newArrayList();

        for (Column column : allColumns) {
            for (Column column_ : columns) {
                if (!column.equals(column_)) {
                    toRemove.add(column);
                }
            }
        }

        Schema newSchema = new Schema(schema);

        for (Column column : toRemove) {
            newSchema.remove(column);
        }

        for (Tuple tuple : tuples) {
            tuple.project(newSchema);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table orderBy(List<Column> columns) {
        Collections.sort(tuples, TupleComparator.of(columns));
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table filter(List<Predicate> expressions) {
        List<Tuple> filtered = Lists.newArrayList();
        for (Predicate expression : expressions) {
            Column column = expression.getLeft();
            Object value = expression.getValue();
            for (Tuple tuple : tuples) {
                String v = tuple.get(column).toString();
                if (!v.equals(value)) {
                    filtered.add(tuple);
                }
            }
        }

        for (Tuple tuple : filtered) {
            tuples.remove(tuple);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Table> groupOn(List<Column> columns) {
        Preconditions.checkNotNull(columns);
        Schema schema = getSchema();

        for (Column column : columns) {
            if (!schema.hasColumn(column)) {
                throw new IllegalArgumentException("Column " + column + "does not exist.");
            }
        }

        HashMap<String, List<Tuple>> map = new HashMap<>();
        for (int i = 0; i < size(); i ++) {
            Tuple tuple = get(i);
            StringBuilder builder = new StringBuilder();
            for (Column column : columns) {
                Object obj = tuple.get(column);
                if (obj != null) {
                    builder.append(obj.toString());
                }
                builder.append("_");
            }

            String hash = builder.toString();
            // TODO: double-check the correctness
            if (map.containsKey(hash)) {
                List<Tuple> tupleList = map.get(hash);
                tupleList.add(tuple);
            } else {
                List<Tuple> newGroup = new ArrayList<>();
                newGroup.add(tuple);
                map.put(hash, newGroup);
            }
        }

        List<Table> lists = new ArrayList<>();
        for (List<Tuple> group : map.values()) {
            MemoryTable newTable = MemoryTable.of(group);
            lists.add(newTable);
        }
        return lists;
    }
}
