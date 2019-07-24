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

import com.google.common.collect.Lists;

import java.util.Comparator;
import java.util.List;

/**
 * Comparator which compares two {@link Tuple} by {@link Column}.
 *
 */
class TupleComparator implements Comparator<Tuple> {
    private List<Column> columns;
    private TupleComparator(List<Column> columns) {
        this.columns = columns;
    }

    /**
     * Factory method to create a TupleComparator.
     * @param column column.
     * @return a TupleComparator.
     */
    public static TupleComparator of(Column column) {
        List<Column> columns = Lists.newArrayList(column);
        return new TupleComparator(columns);
    }

    /**
     * Factory method to create a TupleComparator.
     * @param columns a list of columns.
     * @return a TupleComparator.
     */
    public static TupleComparator of(List<Column> columns) {
        return new TupleComparator(columns);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int compare(Tuple o1, Tuple o2) {
        int result = 0;
        for (Column column : columns) {
            String s1 = o1.get(column).toString();
            String s2 = o2.get(column).toString();
            result = s1.compareTo(s2);
            if (result != 0) {
                break;
            }
        }
        return result;
    }
}