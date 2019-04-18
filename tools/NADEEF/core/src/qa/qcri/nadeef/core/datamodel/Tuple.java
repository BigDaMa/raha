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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import qa.qcri.nadeef.tools.Logger;

import java.nio.charset.Charset;
import java.util.List;
import java.util.ArrayList;


/**
 * Tuple class represents a tuple (row) in a table.
 */
public class Tuple {
    private Logger tracer = Logger.getLogger(Tuple.class);

    //<editor-fold desc="Private Fields">
    private List<byte[]> values;
    private Schema schema;
    private int tid;
    //</editor-fold>

    //<editor-fold desc="Public Members">

    /**
     * Construct a tuple.
     * @param tupleId tuple id.
     * @param schema tuple schema.
     * @param values tuple values.
     */
    public Tuple(int tupleId, Schema schema, List<byte[]> values) {
        if (schema == null || values == null) {
            throw new IllegalArgumentException("Input Schema/Values cannot be null.");
        }

        if (schema.size() != values.size()) {
            throw new IllegalArgumentException(
                "Tuple values does not match the schema. " +
                "Schema has size of " + schema.size() +
                " but values has size of " + values.size()
            );
        }

        if (tupleId < 1) {
            throw new IllegalArgumentException("Tuple ID cannot be less than 1.");
        }

        this.tid = tupleId;
        this.schema = schema;
        this.values = values;
    }

	public List<byte[]> getValues(){
		return values;
	}

	public Tuple cloneObj(){
		List<byte[]> clonedValues = new ArrayList<byte[]>();
		for (byte[] v : this.values){
			clonedValues.add((byte[])v.clone());
		}
		Tuple tuple = new Tuple(tid,schema, clonedValues);
		return tuple;
	}

	
	public void setCell(String columnName, byte[] value){
		Column column = new Column(schema.getTableName(), columnName);
		 int index = schema.get(column);
		 values.set(index, value);
	}

	
    /**
     * Gets the value from the tuple.
     * @param key The attribute key
     * @return Output Value
     */
    public Object get(Column key) {
        int index = schema.get(key);
        byte[] bytes = values.get(index);
        Object result = null;
        if (bytes != null) {
            DataType type = schema.getTypes()[index];
            String stringValue = new String(bytes, Charset.forName("UTF-8"));
            switch (type) {
                case STRING:
                    result = stringValue;
                    break;
                case INTEGER:
                    result = Integer.parseInt(stringValue);
                    break;
                case DOUBLE:
                    result = Double.parseDouble(stringValue);
                    break;
                case FLOAT:
                    result = Float.parseFloat(stringValue);
                    break;
                case BOOL:
                    result = Boolean.parseBoolean(stringValue);
                    break;
                case TIMESTAMP:
                    result = stringValue;
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown data type");
            }
        }
        return result;
    }

    /**
     * Gets the value from the tuple.
     * @param columnName The attribute key
     * @return Output Value
     */
    public Object get(String columnName) {
        Column column = new Column(schema.getTableName(), columnName);
        return get(column);
    }

    /**
     * Gets the Tuple id.
     * @return tuple id.
     */
    public int getTid() {
        return tid;
    }

    /**
     * Gets the Cell given a column key.
     * @param key key.
     * @return Cell.
     */
    public Cell getCell(Column key) {
        return new Cell(key, tid, get(key));
    }

    /**
     * Gets the Cell given a column key.
     * @param key key.
     * @return Cell.
     */
    public Cell getCell(String key) {
        return getCell(new Column(schema.getTableName(), key));
    }

    /**
     * Gets all the values in the tuple.
     * @return value collections.
     */
    public ImmutableSet<Cell> getCells() {
        Column[] columns = schema.getColumns();
        List<Cell> cells = Lists.newArrayList();
        for (Column column : columns) {
            if (column.getColumnName().equalsIgnoreCase("tid")) {
                continue;
            }
            Cell cell = new Cell(column, tid, get(column));
            cells.add(cell);
        }
        return ImmutableSet.copyOf(cells);
    }

    /**
     * Gets all the cells in the tuple.
     * @return Attribute collection
     */
    public Schema getSchema() {
        return schema;
    }

    /**
     * Returns <code>True</code> when the tuple is from the given table name.
     * @param tableName table name.
     * @return <code>True</code> when the tuple is from the given table name.
     */
    public boolean isFromTable(String tableName) {
        String tableName_ = schema.getTableName();
        if (tableName_.equalsIgnoreCase(tableName)) {
            return true;
        }

        if (tableName_.startsWith("TB_")) {
            String originalTableName = tableName_.substring(3);
            return originalTableName.equalsIgnoreCase(tableName);
        }
        return false;
    }

    /**
     * Returns <code>True</code> when given a tuple from the same schema, the values are
     * also the same. There is no check on the schema but only do a check on the values.
     * This is mainly used for optimization on tuple compare from the same schema.
     * @param tuple tuple to compare.
     * @return <code>True</code> when the given tuple from the same schema also has the same
     * values.
     */
    public boolean hasSameValue(Tuple tuple) {
        if (tuple == null) {
            return false;
        }

        if (this == tuple || this.values == tuple.values) {
            return true;
        }

        if (values.size() != tuple.values.size()) {
            return false;
        }

        // Tuples are the same when TID is the same within the same table.
        if (
            tid == tuple.tid &&
            schema.getTableName().equalsIgnoreCase(tuple.schema.getTableName())
        ) {
            return true;
        }

        Optional<Integer> tidIndex = schema.getTidIndex();
        for (int i = 0; i < values.size(); i ++) {
            // skip the TID compare because we know that they are different.
            if (tidIndex.isPresent() && i == tidIndex.get()) {
                continue;
            }

            if (values.get(i) == tuple.values.get(i)) {
                continue;
            }

            if (values.get(i).length != tuple.values.get(i).length) {
                return false;
            }

            for (int j = 0; j < values.get(i).length; j ++) {
                if (values.get(i)[j] != tuple.values.get(i)[j]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Project the Tuple on a new schema.
     * @param newSchema new schema.
     */
    void project(Schema newSchema) {
        Column[] columns = newSchema.getColumns();
        List<byte[]> nvalues = Lists.newArrayList();
        for (Column column : columns) {
            int index = schema.get(column);
            nvalues.add(values.get(index));
        }
        values = nvalues;
        schema = newSchema;
    }
    //</editor-fold>
}
