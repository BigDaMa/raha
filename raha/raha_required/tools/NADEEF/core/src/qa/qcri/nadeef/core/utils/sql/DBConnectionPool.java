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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.apache.commons.dbcp.BasicDataSource;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.PerfReport;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialectTools;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Database JDBC Connection Factory class.
 */
public class DBConnectionPool {
    private static Logger tracer = Logger.getLogger(DBConnectionPool.class);
    private static final int MAX_ACTIVE = Runtime.getRuntime().availableProcessors() * 2;

    private BasicDataSource nadeefPool;
    private BasicDataSource sourcePool;
    private DBConfig sourceConfig;
    private DBConfig nadeefConfig;
    private HashSet<String> localCache;
    private static HashMap<String, String> indexCache = Maps.newHashMap();
    private static HashMap<String, Integer> indexCount = Maps.newHashMap();
    private static Object indexLockObject = new Object();

    private DBConnectionPool(DBConfig sourceConfig, DBConfig nadeefConfig) {
        this.sourceConfig = Preconditions.checkNotNull(sourceConfig);
        this.nadeefConfig = Preconditions.checkNotNull(nadeefConfig);

        nadeefPool = createConnectionPool(nadeefConfig);
        sourcePool = createConnectionPool(sourceConfig);
        localCache = Sets.newHashSet();
    }

    // <editor-fold desc="Public methods">

    /**
     * Creates a connection pool factory.
     * @param source source dbConfig.
     * @param nadeefConfig NADEEF dbConfig.
     * @return connection pool instance.
     */
    public static DBConnectionPool createDBConnectionPool(
        DBConfig source,
        DBConfig nadeefConfig
    ) {
        return new DBConnectionPool(source, nadeefConfig);
    }

    /**
     * Create a connection pool.
     * @param dbconfig input DB config.
     * @return connection pool instance.
     */
    public BasicDataSource createConnectionPool(DBConfig dbconfig) {
        tracer.fine("Creating connection pool for " + dbconfig.getUrl());
        BasicDataSource result;
        result = new BasicDataSource();
        result.setUrl(dbconfig.getUrl());
        result.setDriverClassName(SQLDialectTools.getDriverName(dbconfig.getDialect()));

        String username = dbconfig.getUserName();
        if (!Strings.isNullOrEmpty(username)) {
            result.setUsername(username);
        }

        String password = dbconfig.getPassword();
        if (!Strings.isNullOrEmpty(password)) {
            result.setPassword(password);
        }

        result.setMaxActive(MAX_ACTIVE);
        result.setMaxIdle(MAX_ACTIVE * 3);
        result.setDefaultAutoCommit(false);
        return result;
    }

    /**
     * Shutdown the connection pool.
     */
    public void shutdown() {
        // drop all the indexes.

        if (sourcePool != null) {
            try (
                Connection conn = sourcePool.getConnection();
                Statement stat = conn.createStatement();
            ) {
                synchronized (indexLockObject) {
                    for (String indexName : localCache) {
                        int count =
                            indexCount.containsKey(indexName) ? indexCount.get(indexName) - 1 : 0;

                        // remove index when count goes to 0.
                        if (count == 0) {
                            String tableName = indexCache.get(indexName);
                            SQLDialectBase dialectManager =
                                SQLDialectFactory.getDialectManagerInstance(
                                    sourceConfig.getDialect()
                                );
                            stat.executeUpdate(dialectManager.dropIndex(indexName, tableName));
                            indexCache.remove(indexName);
                            indexCount.remove(indexName);
                        } else {
                            indexCount.put(indexName, count);
                        }
                    }
                    conn.commit();
                }

                localCache.clear();
            } catch (Exception ex) {
                tracer.error("Exceptions happen when closing the connection pool.", ex);
            }
        }

        try {
            if (nadeefPool != null) {
                nadeefPool.close();
                tracer.fine("Closing a connection pool @" + nadeefPool.getUrl());
            }

            if (sourcePool != null) {
                sourcePool.close();
                tracer.fine("Closing a connection pool @" + sourcePool.getUrl());
            }
        } catch (Exception e) {
            tracer.error("Exception during closing NADEEF pool.", e);
        }
    }

    /**
     * Creates a new JDBC connection on the Nadeef database.
     * @return new JDBC connection.
     */
    public Connection getNadeefConnection() throws SQLException {
        PerfReport.addMetric(PerfReport.Metric.NadeefDBConnectionCount, 1);
        return nadeefPool.getConnection();
    }

    /**
     * Creates a new JDBC connection on the source DB from a clean plan.
     * @return new JDBC connection.
     */
    public Connection getSourceConnection() throws SQLException {
        PerfReport.addMetric(PerfReport.Metric.SourceDBConnectionCount, 1);
        return sourcePool.getConnection();
    }

    /**
     * Gets the source {@link DBConfig}.
     * @return {@link DBConfig}.
     */
    public DBConfig getSourceDBConfig() {
        return sourceConfig;
    }

    /**
     * Gets the NADEEF {@link DBConfig}.
     * @return {@link DBConfig}.
     */
    public DBConfig getNadeefConfig() {
        return nadeefConfig;
    }

    public void createIndexIfNotExist(String tableName, String fullColumnName) {
        synchronized (indexLockObject) {
            String indexName = "IDX_" + tableName + "_" + fullColumnName;

            // make sure a pool only counts one for the index.
            if (localCache.contains(indexName)) {
                return;
            }

            localCache.add(indexName);
            if (!indexCache.containsKey(indexName)) {
                Connection conn = null;
                Statement stat = null;

                try {
                    conn = sourcePool.getConnection();
                    stat = conn.createStatement();
                    // create the index.
                    String indexSQL =
                        "CREATE INDEX " + indexName + " ON " +
                            tableName + " (" + fullColumnName + ")";
                    stat.executeUpdate(indexSQL);

                    // in case of creating failure, this will prevent the exception happens again.
                    indexCache.put(indexName, tableName);
                    indexCount.put(indexName, 1);

                    conn.commit();

                    PerfReport.addMetric(PerfReport.Metric.SourceIndexCreationCount, 1);
                } catch (Exception ex) {
                    tracer.error("Creating index " + indexName + " failed.", ex);
                } finally {
                    try {
                        if (conn != null) {
                            conn.close();
                        }

                        if (stat != null) {
                            stat.close();
                        }
                    } catch (Exception ex) {}
                }
            } else {
                int count = indexCount.get(indexName);
                indexCount.put(indexName, count + 1);
            }
        }
    }

    /**
     * Gets the JDBC connection based on the dialect.
     * @param dbConfig dbconfig.
     * @param autoCommit auto commit flag.
     * @return JDBC connection.
     */
    public static Connection createConnection(DBConfig dbConfig, boolean autoCommit)
        throws
            ClassNotFoundException,
            SQLException,
            IllegalAccessException,
            InstantiationException {
        PerfReport.addMetric(PerfReport.Metric.DBConnectionCount, 1);
        String driverName = SQLDialectTools.getDriverName(dbConfig.getDialect());
        Class.forName(driverName).newInstance();
        Connection conn =
            DriverManager.getConnection(
                dbConfig.getUrl(),
                dbConfig.getUserName(),
                dbConfig.getPassword()
            );
        conn.setAutoCommit(autoCommit);
        return conn;
    }

    /**
     * Gets the JDBC connection based on the dialect.
     * @param dbConfig dbconfig.
     * @return JDBC connection.
     */
    public static Connection createConnection(DBConfig dbConfig)
            throws
            ClassNotFoundException,
            SQLException,
            IllegalAccessException,
            InstantiationException {
        return createConnection(dbConfig, false);
    }

    //</editor-fold>
}
