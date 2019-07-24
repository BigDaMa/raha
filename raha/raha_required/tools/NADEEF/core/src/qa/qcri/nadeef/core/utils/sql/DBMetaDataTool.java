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

package qa.qcri.nadeef.core.utils.sql;

import com.google.common.collect.Lists;
import com.mysql.jdbc.log.Log;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.DataType;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.sql.*;
import java.util.List;

/**
 * An utility class for getting meta data from database.
 */
public final class DBMetaDataTool {
    /**
     * Copies the table within the database.
     * @param dbConfig working database config.
     * @param dialectManager Dialect manager.
     * @param sourceTableName source table name.
     * @param targetTableName target table name.
     */
    public static void copy(
        DBConfig dbConfig,
        SQLDialectBase dialectManager,
        String sourceTableName,
        String targetTableName
    ) throws
        ClassNotFoundException,
        SQLException,
        InstantiationException,
        IllegalAccessException {
        Connection conn = null;
        Statement stat = null;
        try {
            conn = DBConnectionPool.createConnection(dbConfig, true);
            stat = conn.createStatement();
            if (isTableExist(dbConfig, targetTableName)) {
                stat.execute(dialectManager.dropTable(targetTableName));
            }
            dialectManager.copyTable(conn, sourceTableName, targetTableName);
        } finally {
            if (stat != null) {
                stat.close();
            }

            if (conn != null) {
                conn.close();
            }
        }
    }

    /**
     * Gets the table schema from source database.
     * @param tableName table name.
     * @return the table schema given a database configuration.
     */
    public static Schema getSchema(DBConfig config, String tableName)
        throws Exception {
        if (!isTableExist(config, tableName)) {
            throw new IllegalArgumentException("Unknown table name " + tableName);
        }

        SQLDialectBase dialectManager =
            SQLDialectFactory.getDialectManagerInstance(config.getDialect());
        Logger tracer = Logger.getLogger(DBMetaDataTool.class);
        Connection conn = null;
        Statement stat = null;
        ResultSet resultSet = null;
        Schema result = null;

        try {
            SQLQueryBuilder builder = new SQLQueryBuilder();
            builder.addFrom(tableName);
            builder.setLimit(1);
            String sql = builder.build(dialectManager);

            conn = DBConnectionPool.createConnection(config);
            stat = conn.createStatement();

            resultSet = stat.executeQuery(sql);
            ResultSetMetaData metaData = resultSet.getMetaData();
            int count = metaData.getColumnCount();
            Column[] columns = new Column[count];
            DataType[] types = new DataType[count];
            for (int i = 1; i <= count; i ++) {
                String attributeName = metaData.getColumnName(i);
                types[i - 1] = DataType.getDataType(metaData.getColumnTypeName(i));
                columns[i - 1] = new Column(tableName, attributeName);
            }

            result = new Schema(tableName, columns, types);
        } catch (Exception ex) {
            tracer.error("Cannot get valid schema.", ex);
        } finally {
            try {
                if (resultSet != null) {
                    resultSet.close();
                }

                if (stat != null) {
                    stat.close();
                }

                if (conn != null) {
                    conn.close();
                }
            } catch (SQLException ex) {}
        }
        return result;
    }

    /**
     * Gets all the tables from the given {@inheritDoc DBConfig}.
     * @param dbConfig DBConfig.
     * @return table string collection.
     */
    public static List<String> getTables(DBConfig dbConfig) throws Exception {
        Connection conn = null;
        ResultSet rs = null;
        List<String> tables = Lists.newArrayList();
        try {
            conn = DBConnectionPool.createConnection(dbConfig, true);
            DatabaseMetaData metaData = conn.getMetaData();
            SQLDialect dialect = dbConfig.getDialect();

            if (dialect == SQLDialect.DERBY || dialect == SQLDialect.DERBYMEMORY) {
                rs = metaData.getTables(
                    null,
                    dbConfig.getUserName().toUpperCase(),
                    null,
                    new String[] { "TABLE" }
                );
            } else {
                rs = metaData.getTables(null, null, null, new String[] { "TABLE" });
            }

            while (rs.next()) {
                tables.add(rs.getString(3));
            }
        } finally {
            try {
                if (rs != null) {
                    rs.close();
                }

                if (conn != null) {
                    conn.close();
                }
            } catch (Exception ex) {}
        }
        return tables;
    }

    /**
     * Returns <code>True</code> when the given table exists in the connection.
     * @param tableName table name.
     * @return <code>True</code> when the given table exists in the connection.
     */
    public static boolean isTableExist(DBConfig dbConfig, String tableName)
            throws
                SQLException,
                IllegalAccessException,
                InstantiationException,
                ClassNotFoundException {
        Connection conn = null;
        try {
            conn = DBConnectionPool.createConnection(dbConfig, true);
            DatabaseMetaData meta = conn.getMetaData();
            boolean result;
            SQLDialect dialect = dbConfig.getDialect();
            if (dialect == SQLDialect.DERBYMEMORY || dialect == SQLDialect.DERBY) {
                String username = dbConfig.getUserName().toUpperCase();
                result =
                    meta.getTables(null, username, tableName, null).next() ||
                    meta.getTables(null, username, tableName.toUpperCase(), null).next() ||
                    meta.getTables(null, username, tableName.toLowerCase(), null).next();
            } else if (dialect == SQLDialect.POSTGRES) {
                result =
                    meta.getTables(null, null, tableName.toUpperCase(), null).next() ||
                    meta.getTables(null, null, tableName.toLowerCase(), null).next();
            } else {
                result = meta.getTables(null, null, tableName, null).next();
            }
            return result;
        } finally {
            if (conn != null) {
                conn.close();
            }
        }
    }

    /**
     * Returns <code>True</code> when the given table exists in the connection.
     * @param tableName table name.
     * @return <code>True</code> when the given table exists in the connection.
     */
    public static int getMaxTid(DBConfig dbConfig, String tableName)
        throws
        SQLException,
        IllegalAccessException,
        InstantiationException,
        ClassNotFoundException {
        SQLDialectBase dialectManager =
            SQLDialectFactory.getDialectManagerInstance(dbConfig.getDialect());
        Logger tracer = Logger.getLogger(DBMetaDataTool.class);
        Connection conn = null;
        Statement stat = null;
        ResultSet resultSet = null;
        int result = 0;
        try {

            conn = DBConnectionPool.createConnection(dbConfig, true);
            stat = conn.createStatement();

            resultSet = stat.executeQuery(dialectManager.selectMaxTid(tableName));
            if (resultSet.next()) {
                result = resultSet.getInt(1);
            }
        } catch (Exception ex) {
            tracer.error("Cannot get valid schema.", ex);
        } finally {
            try {
                if (resultSet != null) {
                    resultSet.close();
                }

                if (stat != null) {
                    stat.close();
                }

                if (conn != null) {
                    conn.close();
                }
            } catch (SQLException ex) {}
        }
        return result;
    }
}
