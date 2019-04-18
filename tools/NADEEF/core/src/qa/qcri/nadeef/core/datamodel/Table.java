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
import com.google.common.base.Strings;
import com.google.common.collect.Lists;

import java.util.Collection;
import java.util.List;

/**
 * A Table represents a collection of {@link Tuple}.
 */
public abstract class Table {
    protected Schema schema;
    protected String tableName;

    //<editor-fold desc="Public methods">
    /**
     * Constructor.
     * @param schema schema.
     */
    public Table(Schema schema) {
        this(Preconditions.checkNotNull(schema).getTableName());
        this.schema = schema;
    }

    /**
     * Constructor.
     */
    public Table(String tableName) {
        Preconditions.checkArgument(!Strings.isNullOrEmpty(tableName));
        this.tableName = tableName;
    }

    /**
     * Gets the Tuple schema.
     * @return Tuple schema.
     */
    public Schema getSchema() {
        return schema;
    }

    /**
     * Gets the size of the collection.
     * @return size of the collection.
     */
    public abstract int size();

    /**
     * Gets the <code>Tuple</code> at the index.
     * @param i index.
     * @return Tuple.
     */
    public abstract Tuple get(int i);

    /**
     * Clean up the resources for this <code>Table</code>. After recycling the table instance
     * should not be used any more.
     */
    public void recycle() {}
    //</editor-fold desc="Public methods">

    //<editor-fold desc="Default Table behavior">

    /**
     * Project the table.
     * @param columnName column name.
     * @return projected table.
     */
    public Table project(String columnName) {
        return project(new Column(tableName, columnName));
    }

    /**
     * Project the table.
     * @param column column.
     * @return projected table.
     */
    public Table project(Column column) {
        return project(Lists.newArrayList(column));
    }

    /**
     * Project the table.
     * @param columns a list of columns.
     * @return projected table.
     */
    public abstract Table project(List<Column> columns);

    /**
     * Sort the table based on a column.
     * @param columnName column name.
     * @return sorted table.
     */
    public Table orderBy(String columnName) {
        return orderBy(new Column(tableName, columnName));
    }

    /**
     * Sort the table based on a column.
     * @param column column.
     * @return sorted table.
     */
    public Table orderBy(Column column) {
        return orderBy(Lists.newArrayList(column));
    }

    /**
     * Sort the table based on a list of columns.
     * @param columns a list of columns.
     * @return sorted table.
     */
    public abstract Table orderBy(List<Column> columns);

    /**
     * Filter the table based on a {@link Predicate}.
     * @param expression expression class.
     * @return filtered table.
     */
    public Table filter(Predicate expression) {
        return filter(Lists.newArrayList(expression));
    }

    /**
     * Filter the table based on a list of {@link Predicate}.
     * @param expressions a list of expressions.
     * @return filtered table.
     */
    public abstract Table filter(List<Predicate> expressions);

    /**
     * Partition the Table into multiple tuple Table based on a list of columns.
     * @param columns Partitions based on a list of column.
     * @return A collection of tuple collections.
     */
    public abstract Collection<Table> groupOn(List<Column> columns);

    /**
     * Partition the table into multiple table based
     * on the column.
     * @param columnName Partition based on column attribute name.
     * @return A collection of tuple collections.
     */
    public Collection<Table> groupOn(String columnName) {
        return groupOn(new Column(getSchema().getTableName(), columnName));
    }

    /**
     * Partition the table into multiple table based
     * on the column.
     * @param column Partition based on column.
     * @return A collection of tuple collections.
     */
    public Collection<Table> groupOn(Column column) {
        return groupOn(Lists.newArrayList(column));
    }
    //</editor-fold desc="Default Table behavior">
}
