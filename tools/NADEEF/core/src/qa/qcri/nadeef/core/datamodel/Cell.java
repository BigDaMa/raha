/**
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

/**
 * A Cell represents a basic unit in a data table. A Cell
 * contains a <code>Column</code> and a Value with an unique TupleId.
 */
public class Cell {
    //<editor-fold desc="Private fields">
    private int tid;
    private Column column;
    private Object value;
    //</editor-fold>

    /**
     * Cell builder class.
     */
    public static class Builder {
        private int tupleId;
        private Column column;
        private Object value;

        public Builder() {}

        public Builder tid(int tid) {
            this.tupleId = tid;
            return this;
        }

        public Builder column(Column column) {
            this.column = column;
            return this;
        }

        public Builder column(String column) {
            this.column = new Column(column);
            return this;
        }

        public Builder value(Object value) {
            this.value = value;
            return this;
        }

        public Cell build() {
            return new Cell(column, tupleId, value);
        }
    }

    //<editor-fold desc="Constructors">
    /**
     * Constructor.
     * @param column Column.
     * @param tid TupleId.
     * @param value Value.
     */
    public Cell(Column column, int tid, Object value) {
        this.column = column;
        this.value = value;
        this.tid = tid;
    }
    //</editor-fold>

    /**
     * Sets the column value. 
     * @return void.
     */
    public void setValue(Object obj) {
        value = obj;
    }

    //<editor-fold desc="Getter and Setters">
    /**
     * Gets the column.
     * @return column.
     */
    public Column getColumn() {
        return column;
    }

    /**
     * Gets the column value. It does the type inferring and throws exception when
     * value is not the correct type.
     * @return column value.
     */
    @SuppressWarnings("unchecked")
    public <T> T getValue() {
        return (T)value;
    }

    /**
     * Gets Tuple Id.
     * @return tuple id.
     */
    public int getTid() {
        return tid;
    }

    /**
     * Returns <code>True</code> when the Cell has the same column name as the input.
     * @param columnName column name.
     * @return <code>True</code> when the Cell has the same column name as the input.
     */
    public boolean hasColumnName(String columnName) {
        return column.getColumnName().equalsIgnoreCase(columnName);
    }

    //</editor-fold>

    //<editor-fold desc="Override Equals and Hash code">

    /**
     * Returns <code>True</code> when the given object is as the same as the input.
     * @param obj input object.
     * @return <code>True</code> when the given object is as the same as the input.
     */
    @Override
    public boolean equals(Object obj) {
        if (obj == null || !(obj instanceof Cell)) {
            return false;
        }

        Cell row = (Cell)obj;

        return row.column.equals(column) && row.tid == tid;
    }

    /**
     * Returns the hash code of the column.
     * @return the hash code of the column.
     */
    @Override
    public int hashCode() {
        return column.hashCode() * tid;
    }
    //</editor-fold>
}
