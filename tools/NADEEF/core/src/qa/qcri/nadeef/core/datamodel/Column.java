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

import qa.qcri.nadeef.tools.CommonTools;

/**
 * A Column represents a Column in a table. It contains a table name and a attribute name.
 *
 */
public class Column {
    private String tableName;
    private String columnName;
    private String schemaName;

    //<editor-fold desc="Constructors">
    /**
     * Constructor.
     */
    public Column(String tableName, String columnName) {
        this("public", tableName, columnName);
    }

    /**
     * Constructor.
     */
    public Column(String fullAttributeName) {
        if (!CommonTools.isValidColumnName(fullAttributeName)) {
            throw new IllegalArgumentException("Invalid full attribute name " + fullAttributeName);
        }

        String[] splits = fullAttributeName.split("\\.");

        schemaName = "public";
        tableName = splits[0];
        columnName = splits[1];
    }

    /**
     * Constructor.
     * @param schemaName schema.
     * @param tableName table.
     * @param columnName attribute.
     */
    public Column(String schemaName, String tableName, String columnName) {
        if (columnName == null || tableName == null || schemaName == null) {
            throw new IllegalArgumentException("Attribute name cannot be null.");
        }
        this.schemaName = schemaName;
        this.tableName = tableName;
        this.columnName = columnName;
    }
    //</editor-fold>

    /**
     * Gets the table name.
     * @return original table name.
     */
    public String getTableName() {
        return tableName;
    }

    /**
     * Returns <code>True</code> when the tuple is from the given table name.
     * @param tableName table name.
     * @return <code>True</code> when the tuple is from the given table name.
     */
    public boolean isFromTable(String tableName) {
        if (this.tableName.equalsIgnoreCase(tableName)) {
            return true;
        }

        if (this.tableName.startsWith("TB_")) {
            String originalTableName = this.tableName.substring(3);
            return originalTableName.equalsIgnoreCase(tableName);
        }
        return false;
    }

    /**
     * Gets the column name.
     * @return column name.
     */
    public String getColumnName() {
        return columnName;
    }

    /**
     * Gets a string with format of 'tableName'.'columnName'.
     */
    public String getFullColumnName() {
        return getTableName() + "." + columnName;
    }

    //<editor-fold desc="Custom equal / hashcode">
    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }

        if (obj == null || !(obj instanceof Column)) {
            return false;
        }

        Column column = (Column)obj;

        return getFullColumnName().equalsIgnoreCase(column.getFullColumnName());
    }

    @Override
    public int hashCode() {
        final int root = 109;
        return
            root * getTableName().toLowerCase().hashCode() *
                columnName.toLowerCase().hashCode() *
                schemaName.toLowerCase().hashCode();
    }
    //</editor-fold>

}