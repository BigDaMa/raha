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
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.core.utils.sql.SQLQueryBuilder;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;

import java.io.IOException;
import java.nio.charset.Charset;
import java.sql.*;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * SQLTable represents a {@link Table} which resides in a database.
 */
public class SQLTable extends Table {
    private static Logger tracer = Logger.getLogger(SQLTable.class);

    private DBConnectionPool connectionFactory;
    private SQLDialectBase dialectManager;
    private String tableName;
    private SQLQueryBuilder sqlQuery;
    private List<Tuple> tuples;
    private long updateTimestamp = -1;
    private long changeTimestamp = System.currentTimeMillis();
    private Object lock;

    //<editor-fold desc="Constructor">
    /**
     * Constructor with database connection.
     * @param tableName tuple collection table name.
     * @param connectionFactory database connection pool.
     */
    public SQLTable(String tableName, DBConnectionPool connectionFactory) {
        super(tableName);
        this.connectionFactory = connectionFactory;
        this.dialectManager =
            SQLDialectFactory.getDialectManagerInstance(
                connectionFactory.getSourceDBConfig().getDialect()
            );
        this.tableName = tableName;
        this.sqlQuery = new SQLQueryBuilder();
        this.sqlQuery.addFrom(tableName);
        this.lock = new Object();
    }

    //</editor-fold>

    //<editor-fold desc="Table Interface">
    /**
     * Gets the size of the collection.
     * It will call <code>syncData</code> if the collection is not yet existed.
     *
     * @return size of the collection.
     */
    @Override
    public int size() {
        syncDataIfNeeded();
        return tuples.size();
    }

    /**
     * Gets the schema of the Table.
     * @return the schema.
     */
    @Override
    public Schema getSchema() {
        if (schema == null) {
            synchronized (lock) {
                if (schema == null)
                    syncSchema();
            }
        }

        return schema;
    }

    /**
     * Gets the tuple from the collection.
     * @param i tuple index.
     * @return tuple instance.
     */
    @Override
    public Tuple get(int i) {
        syncDataIfNeeded();
        return tuples.get(i);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table project(List<Column> columns) {
        Preconditions.checkNotNull(columns);
        for (Column column : columns) {
            sqlQuery.addSelect(column.getColumnName());
        }

        synchronized (lock) {
            changeTimestamp = System.currentTimeMillis();
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table orderBy(List<Column> columns) {
        for (Column column : columns) {
            sqlQuery.addOrder(column.getColumnName());
        }

        synchronized (lock) {
            changeTimestamp = System.currentTimeMillis();
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Table filter(List<Predicate> expressions) {
        for (Predicate expression : expressions) {
            sqlQuery.addWhere(expression.toSQLString());
        }

        synchronized (lock) {
            changeTimestamp = System.currentTimeMillis();
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override public Collection<Table> groupOn(List<Column> columns) {
        // TODO: check memory constraints
        this.syncDataIfNeeded();
        MemoryTable table = MemoryTable.of(this.tuples);
        return table.groupOn(columns);
    }

    /**
     * Group on operation with memory constraints.
     */
    public Collection<Table> groupOnConstrained(List<Column> columns) {
        Collection<Table> result = Lists.newArrayList();
        ResultSet distinctResult = null;

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < columns.size(); i ++) {
            if (i > 0)
                builder.append(",");
            String columnName = columns.get(i).getColumnName();
            builder.append(columnName);
            connectionFactory.createIndexIfNotExist(tableName, columnName);
        }
        String sql = String.format("SELECT DISTINCT %s FROM %s", builder.toString(), tableName);

        try (Connection conn = connectionFactory.getSourceConnection();
             Statement stat = conn.createStatement()) {
            // create index ad-hoc.
            stat.setFetchSize(8192);
            distinctResult = stat.executeQuery(sql);

            while (distinctResult.next()) {
                SQLTable newTable =
                    new SQLTable(tableName, connectionFactory);
                newTable.sqlQuery = new SQLQueryBuilder(sqlQuery);
                for (int i = 0; i < columns.size(); i ++) {
                    Column column = columns.get(i);
                    Object value = distinctResult.getObject(i + 1);
                    Predicate columnFilter =
                        new Predicate.PredicateBuilder()
                            .left(column)
                            .isSingle()
                            .constant(value)
                            .op(Operation.EQ).build();
                    newTable.sqlQuery.addWhere(columnFilter.toSQLString());
                }
                result.add(newTable);
            }
        } catch (Exception ex) {
            tracer.error(ex.getMessage(), ex);
        } finally {
            if (distinctResult != null) {
                try {
                    distinctResult.close();
                } catch (Exception ex) {}
            }
        }
        return result;
    }

    //</editor-fold>

    //<editor-fold desc="Equalization Interface">
    /**
     * Custom equals compare.
     * @param collection target collection.
     * @return Returns <code>True</code> if the given collection is the same.
     */
    @Override
    public boolean equals(Object collection) {
        if (collection == this) {
            return true;
        }

        if (collection == null || !(collection instanceof SQLTable)) {
            return false;
        }

        SQLTable obj = (SQLTable)collection;
        DBConfig dbConfig1 = connectionFactory.getSourceDBConfig();
        DBConfig dbConfig2 = obj.connectionFactory.getSourceDBConfig();
        if (dbConfig1.equals(dbConfig2) && tableName.equals(obj.tableName)) {
            return true;
        }

        return false;
    }

    /**
     * Calculates the hash code of the <code>Table</code>.
     * @return hash code.
     */
    @Override
    public int hashCode() {
        DBConfig dbconfig = connectionFactory.getSourceDBConfig();
        return dbconfig.hashCode() * tableName.hashCode();
    }

    //</editor-fold>

    //<editor-fold desc="Private members">

    /**
     * Synchronize the data schema with underneath database.
     */
    private void syncSchema() {
        SQLQueryBuilder builder = new SQLQueryBuilder(sqlQuery);
        builder.setLimit(1);
        String sql = builder.build(dialectManager);

        try (
            Connection conn = connectionFactory.getSourceConnection();
            Statement stat = conn.createStatement();
            ResultSet resultSet = stat.executeQuery(sql);
        ) {
            // tracer.verbose(sql);
            ResultSetMetaData metaData = resultSet.getMetaData();
            int count = metaData.getColumnCount();
            Column[] columns = new Column[count];
            DataType[] types = new DataType[count];
            for (int i = 1; i <= count; i ++) {
                String attributeName = metaData.getColumnName(i);
                columns[i - 1] = new Column(tableName, attributeName);
                types[i - 1] = DataType.getDataType(metaData.getColumnTypeName(i));
            }

            schema = new Schema(tableName, columns, types);
        } catch (Exception ex) {
            tracer.error("Cannot get valid schema.", ex);
        }
    }

    /**
     * Synchronize the collection data with the underlying database.
     * @return Returns <code>True</code> when the synchronization is successful.
     */
    private boolean syncData() {
        Stopwatch stopwatch = Stopwatch.createStarted();
        // prepare for the SQL
        String sql = sqlQuery.build(dialectManager);
        ResultSet resultSet = null;
        try (
            Connection conn = connectionFactory.getSourceConnection();
            // get the connection and run the SQL
            Statement stat = conn.createStatement();
        ) {
            // tracer.verbose(sql);
            stat.setFetchSize(4096);
            resultSet = stat.executeQuery(sql);

            // fill the schema
            ResultSetMetaData metaData = resultSet.getMetaData();
            int count = metaData.getColumnCount();
            int tidIndex = 0;
            Column[] columns = new Column[count];
            DataType[] types = new DataType[count];
            for (int i = 1; i <= count; i ++) {
                String columnName = metaData.getColumnName(i);
                columns[i - 1] = new Column(tableName, columnName);
                types[i - 1] = DataType.getDataType(metaData.getColumnTypeName(i));
                if (tidIndex == 0 && columnName.equalsIgnoreCase("tid")) {
                    tidIndex = i;
                }
            }
            schema  = new Schema(tableName, columns, types);

            // fill the tuples
            tuples = Lists.newArrayList();
            int tupleId = -1;
            while (resultSet.next()) {
                List<byte[]> values = Lists.newArrayList();
                if (tidIndex != 0) {
                    tupleId = resultSet.getInt(tidIndex);
                } else {
                    tracer.info("Table does not have an TID column, use 1 as default.");
                    tupleId = 1;
                }

                for (int i = 1; i <= count; i ++) {
                    Object object = resultSet.getObject(i);
                    values.add(serialize(object));
                }

                tuples.add(new Tuple(tupleId, schema, values));
            }
        } catch (Exception ex) {
            tracer.error("Synchronization failed.", ex);
        } finally {
            try {
                if (resultSet != null) {
                    resultSet.close();
                }
            } catch (Exception ex) {}
        }

        PerfReport.addMetric(
            PerfReport.Metric.DBLoadTime,
            stopwatch.elapsed(TimeUnit.MILLISECONDS)
        );
        stopwatch.stop();
        return true;
    }

    private void syncDataIfNeeded() {
        if (updateTimestamp < changeTimestamp) {
            synchronized (lock) {
                syncData();
                updateTimestamp = changeTimestamp;
            }
        }
    }

    /**
     * Serialize object to a bytes array.
     */
    private static byte[] serialize(Object obj) throws IOException {
        byte[] result = null;
        String stringValue;
        if (obj != null) {
            if (obj instanceof String) {
                stringValue = (String)obj;
            } else {
                stringValue = obj.toString();
            }
            // TODO: We currently use hardcoded UTF-8 encoding.
            result = stringValue.getBytes(Charset.forName("UTF-8"));
        }
        return result;
    }
    //</editor-fold>

    //<editor-fold desc="Finalization methods">
    public void recycle() {
        tuples.clear();
        tuples = null;
        tableName = null;
        updateTimestamp = Long.MAX_VALUE;
        connectionFactory = null;
    }

    //</editor-fold>

}
