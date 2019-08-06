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

import java.sql.Types;

/**
 * Primitive data types used in abstract rule.
 */
public enum DataType {
    STRING(0),
    INTEGER(1),
    DOUBLE(2),
    FLOAT(3),
    BOOL(4),
    TIMESTAMP(5);

    private final int value;
    private DataType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    // this is a bug in MySQL JDBC where getType doesn't match the sql.Types
    public static DataType getDataType(String sqlTypeValue) {
        DataType result;
        if (sqlTypeValue.equalsIgnoreCase("serial"))
            result = DataType.INTEGER;
        else if (sqlTypeValue.equalsIgnoreCase("int4"))
            result = DataType.INTEGER;
        else if (sqlTypeValue.equalsIgnoreCase("int"))
            result = DataType.INTEGER;
        else if (sqlTypeValue.equalsIgnoreCase("integer"))
            result = DataType.INTEGER;
        else if (sqlTypeValue.equalsIgnoreCase("varchar"))
            result = DataType.STRING;
        else if (sqlTypeValue.equalsIgnoreCase("text"))
            result = DataType.STRING;
        else if (sqlTypeValue.equalsIgnoreCase("float"))
            result = DataType.FLOAT;
        else if (sqlTypeValue.equalsIgnoreCase("float8"))
            result = DataType.FLOAT;
        else if (sqlTypeValue.equalsIgnoreCase("double"))
            result = DataType.DOUBLE;
        else if (sqlTypeValue.equalsIgnoreCase("float4"))
            result = DataType.DOUBLE;
        else if (sqlTypeValue.equalsIgnoreCase("bool"))
            result = DataType.BOOL;
        else if (sqlTypeValue.equalsIgnoreCase("timestamp"))
            result = DataType.TIMESTAMP;
        else
            throw new IllegalArgumentException("Unknown data types " + sqlTypeValue);
        return result;
    }

    public static DataType getDataType(int sqlTypeValue) {
        DataType result;
        switch (sqlTypeValue) {
            case Types.INTEGER:
                result = DataType.INTEGER;
                break;
            case Types.VARCHAR:
            case Types.NVARCHAR:
                result = DataType.STRING;
                break;
            case Types.FLOAT:
                result = DataType.FLOAT;
                break;
            case Types.DOUBLE:
                result = DataType.DOUBLE;
                break;
            case Types.BOOLEAN:
                result = DataType.BOOL;
                break;
            case Types.TIMESTAMP:
                result = DataType.TIMESTAMP;
                break;
            default:
                throw new IllegalArgumentException("Unknown data types " + sqlTypeValue);
        }
        return result;
    }
}
