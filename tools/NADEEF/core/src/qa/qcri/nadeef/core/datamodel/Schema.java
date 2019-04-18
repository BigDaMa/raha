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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import java.util.List;

/**
 * Schema class provides a mapping between column and value for a table.
 */
public class Schema {
    private String tableName;
    private Column[] columns;
    private DataType[] types;

    //<editor-fold desc="Builder">
    /**
     * Builder class.
     */
    public static class Builder {
        private String tableName;
        List<Column> columns;
        List<DataType> types;

        public Builder() {
            columns = Lists.newArrayList();
            types = Lists.newArrayList();
        }

        public Builder table(String tableName) {
            this.tableName = tableName;
            return this;
        }

        public Builder column(Column columnName, DataType type) {
            columns.add(columnName);
            types.add(type);
            return this;
        }

        public Builder column(String columnName, int value) {
            columns.add(new Column(tableName, columnName));
            types.add(DataType.getDataType(value));
            return this;
        }

        public Builder column(String columnName, DataType type) {
            columns.add(new Column(tableName, columnName));
            types.add(type);
            return this;
        }

        public Schema build() {
            Column[] columns_ = new Column[columns.size()];
            DataType[] types_ = new DataType[types.size()];
            columns.toArray(columns_);
            types.toArray(types_);
            return new Schema(tableName, columns_, types_);
        }
    }
    //</editor-fold>

    /**
     * Constructor.
     * @param tableName table name.
     * @param columns column array.
     */
    public Schema(String tableName, Column[] columns, DataType[] types) {
        this.tableName = Preconditions.checkNotNull(tableName);
        Preconditions.checkArgument(columns != null && columns.length > 0);
        this.columns = columns;
        this.types = types;
    }

    /**
     * Copy constructor.
     * @param schema input schema.
     */
    public Schema(Schema schema) {
        Preconditions.checkNotNull(schema);
        this.tableName = schema.tableName;
        this.columns = schema.getColumns().clone();
        this.types = schema.getTypes().clone();
    }

    /**
     * The size of the schema.
      * @return size.
     */
    public int size() {
        return columns.length;
    }

    /**
     * Returns <code>True</code> when the map contains the column.
     * @param column column.
     * @return <code>True</code> when the map contains the column.
     */
    public boolean hasColumn(Column column) {
        for (Column column_ : columns) {
            if (column_.equals(column)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Gets the table name.
     * @return table name.
     */
    public String getTableName() {
        return tableName;
    }

    /**
     * Gets the column collection.
     * @return column collection.
     */
    public Column[] getColumns() {
        return columns;
    }

    /**
     * Gets the column type collection.
     * @return column type collection.
     */
    public DataType[] getTypes() {
        return types;
    }

    /**
     * Gets the column type collection.
     * @return column type collection.
     */
    public DataType getType(Column column) {
        for (int i = 0; i < columns.length; i ++) {
            if (columns[i].equals(column)) {
                return types[i];
            }
        }
        throw new IllegalArgumentException("Cannot find the column.");
    }

    /**
     * Gets the index from the column.
     * @param column input column.
     * @return Get the index from column.
     */
    public int get(Column column) {
        for (int i = 0; i < columns.length; i ++) {
            if (columns[i].equals(column)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Cannot find the column.");
    }

    /**
     * Returns the TID index of the schema. It returns absent when there is no TID column.
     * @return Returns the TID index of the schema. It returns absent when there is no TID column.
     */
    public Optional<Integer> getTidIndex() {
        for (int i = 0; i < columns.length; i ++) {
            if (columns[i].getColumnName().equalsIgnoreCase("tid")) {
                return Optional.of(i);
            }
        }
        return Optional.absent();
    }

    /**
     * Removes one column.
     * @param column column to be removed.
     */
    void remove(Column column) {
        int result = get(column);
        List<Column> tmp = Lists.newArrayList();
        for (int i = 0; i < columns.length; i ++) {
            if (i != result) {
                tmp.add(columns[i]);
            }
        }
        columns = new Column[tmp.size()];
        tmp.toArray(columns);
    }
}
