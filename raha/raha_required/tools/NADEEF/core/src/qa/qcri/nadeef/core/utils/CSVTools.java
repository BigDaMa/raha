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

package qa.qcri.nadeef.core.utils;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.io.Files;

import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.DBMetaDataTool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * CSVTools is a simple tool which dumps CSV data into database given a table name.
 */
public class CSVTools {
    private static Logger logger = Logger.getLogger(CSVTools.class);
    // <editor-fold desc="Public methods">

    /**
     * Reads the content from CSV file.
     * @param file CSV file.
     * @param separator separator.
     * @return a list of tokens (the header line is skipped).
     */
    public static List<String[]> read(File file, String separator) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        List<String[]> result = Lists.newArrayList();
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            if (Strings.isNullOrEmpty(line)) {
                continue;
            }

            count ++;
            // skip the header
            if (count == 1) {
                continue;
            }

            String[] tokens = line.split(separator);
            result.add(tokens);
        }
        return result;
    }

    /**
     * Append CSV file content into a database table.
     * @param tableName target table name.
     * @param dbConfig DB connection config.
     * @param file CSV file.
     * @return new created table name.
     */
    public static HashSet<Integer> append(
        DBConfig dbConfig,
        SQLDialectBase dialectManager,
        String tableName,
        File file
    ) throws Exception {
        Preconditions.checkNotNull(dbConfig);
        Preconditions.checkNotNull(dialectManager);

        Stopwatch stopwatch = Stopwatch.createStarted();
        HashSet<Integer> result = Sets.newHashSet();
        try {
            boolean hasTableExist = DBMetaDataTool.isTableExist(dbConfig, tableName);

            // Create table
            if (!hasTableExist) {
                throw new IllegalArgumentException("Table " + tableName + " does not exist.");
            }

            // get the current max tid.
            int startTid = DBMetaDataTool.getMaxTid(dbConfig, tableName) + 1;

            // load the data
            int size = 0;

            if (dialectManager.supportBulkLoad()) {
                size = dialectManager.bulkLoad(dbConfig, tableName, file.toPath(), true);
            } else {
                size = dialectManager.fallbackLoad(dbConfig, tableName, file, true);
            }

            logger.info(
                "Appended " + size + " bytes in " +
                stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
            );
            stopwatch.stop();

            // build the tid set.
            int endTid = DBMetaDataTool.getMaxTid(dbConfig, tableName);
            for (int i = startTid; i <= endTid; i ++) {
                result.add(i);
            }

        } catch (Exception ex) {
            logger.error("Cannot load file " + file.getName(), ex);
        }
        return result;
    }

    /**
     * Dumps CSV file content into a database with default schema name and generated table name.
     * @param dbConfig DB connection config.
     * @param dialectManager SQL dialect manager.
     * @param file CSV file.
     * @return new created table name.
     */
    public static String dump(DBConfig dbConfig, SQLDialectBase dialectManager, File file)
            throws IllegalAccessException, SQLException, IOException {
        String fileName = Files.getNameWithoutExtension(file.getName());

        return dump(dbConfig, dialectManager, file, fileName, true);
    }

    public static String dump(
        final DBConfig dbConfig,
        final SQLDialectBase dialectManager,
        final File file,
        final String tableName,
        final boolean overwrite
    ) throws SQLException {
        String header = null;
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            do {
                header = reader.readLine();
            } while (Strings.isNullOrEmpty(header));
        } catch (Exception ex) {
            logger.error("Reading CSV file header failed.", ex);
        }
        return dump(dbConfig, dialectManager, file, tableName, header, overwrite);
    }

    /**
     * Dumps CSV file content into a specified database. It replaces the table if the table
     * already existed.
     * @param dbConfig JDBC connection config.
     * @param file CSV file.
     * @param dialectManager SQL dialect manager.
     * @param tableName new created table name.
     * @param overwrite it overwrites existing table if it exists.
     *
     * @return new created table name.
     */
    public static String dump(
        final DBConfig dbConfig,
        final SQLDialectBase dialectManager,
        final File file,
        final String tableName,
        final String schema,
        final boolean overwrite
    ) throws SQLException {
        Preconditions.checkNotNull(dbConfig);
        Preconditions.checkNotNull(dialectManager);
        Preconditions.checkArgument(!Strings.isNullOrEmpty(schema));

        Stopwatch stopwatch = Stopwatch.createStarted();
        String fullTableName = null;
        String sql;
        try (BufferedReader reader = new BufferedReader(new FileReader(file))){
            // overwrites existing tables if necessary
            fullTableName = "TB_" + tableName;

            boolean hasTableExist = DBMetaDataTool.isTableExist(dbConfig, fullTableName);

            // Create table
            if (hasTableExist && !overwrite) {
                logger.info(
                    "Found table " + fullTableName + " exists and choose not to overwrite."
                );
                return fullTableName;
            } else {
                Statement stat = null;
                try (Connection conn = DBConnectionPool.createConnection(dbConfig, true)) {
                    stat = conn.createStatement();
                    if (hasTableExist && overwrite) {
                        sql = dialectManager.dropTable(fullTableName);
                        logger.fine(sql);
                        stat.execute(sql);
                    }

                    sql = dialectManager.createTableFromCSV(fullTableName, schema);
                    logger.fine(sql);
                    stat.execute(sql);
                    logger.info("Successfully created table " + fullTableName);
                } finally {
                    if (stat != null) {
                        stat.close();
                    }
                }

                // load the data
                int size = 0;
                if (dialectManager.supportBulkLoad()) {
                    size =
                        dialectManager.bulkLoad(
                            dbConfig,
                            fullTableName,
                            file.toPath(),
                            true
                        );
                } else {
                    size = dialectManager.fallbackLoad(dbConfig, fullTableName, file, true);
                }

                logger.info(
                    "Dumped " + size + " bytes in " +
                    stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
                );
                stopwatch.stop();
            }
        } catch (Exception ex) {
            logger.error("Cannot load file " + file.getName(), ex);
        }
        return fullTableName;
    }
    // </editor-fold>
}
