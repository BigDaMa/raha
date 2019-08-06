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
import com.google.common.base.Stopwatch;
import com.google.common.base.Strings;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;
import qa.qcri.nadeef.tools.CommonTools;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Path;
import java.sql.*;
import java.util.concurrent.TimeUnit;

/**
 * Interface for cross vendor Database methods.
 */
public abstract class SQLDialectBase {
    /**
     * Creates SQLDialect instance.
     * @param dialect dialect.
     * @return SQLDialectBase instance.
     */
    public static SQLDialectBase createDialectBaseInstance(SQLDialect dialect) {
        SQLDialectBase dialectInstance;
        switch (dialect) {
            default:
            case DERBYMEMORY:
            case DERBY:
                dialectInstance = new DerbySQLDialect();
                break;
            case POSTGRES:
                dialectInstance = new PostgresSQLDialect();
                break;
            case MYSQL:
                dialectInstance = new MySQLDialect();
                break;
        }
        return dialectInstance;
    }

    /**
     * Returns True when bulk loading is supported.
     * @return True when bulk loading is supported.
     */
    public boolean supportBulkLoad() {
        return false;
    }

    /**
     * Bulk load CSV file.
     * @param dbConfig DBConfig.
     * @param tableName table name.
     * @param file CSV file.
     * @param skipHeader has header.
     * @return line of rows loaded.
     */
    public int bulkLoad(
        DBConfig dbConfig,
        String tableName,
        Path file,
        boolean skipHeader
    ) {
        throw new UnsupportedOperationException("Method is not implemented.");
    }

    /**
     * Loads CSV file when bulk load is not used.
     * @param dbConfig {@link qa.qcri.nadeef.tools.DBConfig}
     * @param tableName table name.
     * @param file CSV file.
     * @param skipHeader skip header.
     * @return line of rows loaded.
     */
    public int fallbackLoad(DBConfig dbConfig, String tableName, File file, boolean skipHeader) {
        Logger tracer = Logger.getLogger(SQLDialectBase.class);
        Stopwatch stopwatch = Stopwatch.createStarted();

        try (
            Connection conn = DBConnectionPool.createConnection(dbConfig, false);
            Statement stat = conn.createStatement();
            BufferedReader reader = new BufferedReader(new FileReader(file))
        ) {
            ResultSet rs = stat.executeQuery(this.selectAll(tableName));
            ResultSetMetaData metaData = rs.getMetaData();
            String line;
            int lineCount = 0;
            int size = 0;
            // Batch load the data
            while ((line = reader.readLine()) != null) {
                lineCount ++;
                if (Strings.isNullOrEmpty(line))
                    continue;
                if (skipHeader && lineCount == 1)
                    continue;
                size += line.toCharArray().length;
                String[] tokens = line.split(",");
                String[] newTokens = new String[tokens.length];
                for (int i = 0; i < tokens.length; i ++) {
                    newTokens[i] = CommonTools.unescapeString(tokens[i], CommonTools.DOUBLE_QUOTE);
                }

                String sql = this.importFromCSV(metaData, tableName, newTokens);
                stat.addBatch(sql);

                if (lineCount % 10240 == 0) {
                    stat.executeBatch();
                }
            }

            stat.executeBatch();
            conn.commit();

            tracer.info(
                "Dumped " + size + " bytes in " +
                stopwatch.elapsed(TimeUnit.MILLISECONDS) + " ms."
            );
            stopwatch.stop();
            return lineCount;
        } catch (Exception ex) {
            tracer.error("Cannot load file " + file.getName(), ex);
        }
        return 0;
    }

    /**
     * Gets the template file.
     * @return template group file.
     */
    protected abstract STGroupFile getTemplate();

    /**
     * Install violation tables.
     * @param violationTableName violation table name.
     * @return SQL statement.
     */
    public String createViolationTable(String violationTableName) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST st = template.getInstanceOf("InstallViolationTable");
        st.add("violationTableName", violationTableName.toUpperCase());
        return st.render();
    }

    /**
     * Install repair tables.
     * @param repairTableName repair table name.
     * @return SQL statement.
     */
    public String createRepairTable(String repairTableName) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST st = template.getInstanceOf("InstallRepairTable");
        st.add("repairTableName", repairTableName.toUpperCase());
        return st.render();
    }

    /**
     * Install auditing tables.
     * @param auditTableName audit table name.
     * @return SQL statement.
     */
    public String createAuditTable(String auditTableName) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST st = template.getInstanceOf("InstallAuditTable");
        st.add("auditTableName", auditTableName.toUpperCase());
        return st.render();
    }

    /**
     * Next Vid.
     * @param tableName violation table name.
     * @return SQL statement.
     */
    public String nextVid(String tableName) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST st = template.getInstanceOf("NextVid");
        st.add("tableName", tableName.toUpperCase());
        return st.render();
    }

    /**
     * Creates a table in the database from a CSV file header.
     * @param tableName table name.
     * @param content table description.
     * @return SQL statement.
     */
    public String createTableFromCSV(String tableName, String content) {
        STGroupFile template = Preconditions.checkNotNull(getTemplate());
        ST st = template.getInstanceOf("CreateTableFromCSV");
        st.add("tableName", tableName.toUpperCase());
        // TODO: remove
        st.add("content", content.replaceAll("string", "varchar(8192)"));
        // st.add("content", content);
        return st.render();
    }

    /**
     * Copy table.
     * @param conn connection.
     * @param sourceName source name.
     * @param targetName target name.
     */
    public abstract void copyTable(
        Connection conn,
        String sourceName,
        String targetName
    ) throws SQLException;

    /**
     * Drop table.
     * @param tableName drop table name.
     * @return SQL statement.
     */
    public String dropTable(String tableName) {
        return "DROP TABLE " + tableName;
    }

    /**
     * Drop index.
     * @param indexName index name.
     * @param tableName drop table name.
     * @return SQL statement.
     */
    public String dropIndex(String indexName, String tableName) {
        return "DROP INDEX " + indexName;
    }

    /**
     * Select star..
     * @param tableName table name.
     * @return SQL statement.
     */
    public String selectAll(String tableName) {
        return "SELECT * FROM " + tableName;
    }

    public String selectMaxTid(String tableName) {
        return "SELECT MAX(TID) FROM " + tableName;
    }

    public String deleteAll(String tableName) {
        return "DELETE FROM " + tableName;
    }

    /**
     * Count the number of rows in the table.
     * @param tableName table name.
     * @return SQL statement.
     */
    public abstract String countTable(String tableName);

    /**
     * Limits the select.
     * @param row row number.
     * @return SQL statement.
     */
    public abstract String limitRow(int row);

    /**
     * Inserts values into a table from CSV row used for batch loading.
     * @param metaData column meta data.
     * @param tableName target table name.
     * @param tokens row values.
     * @return SQL statement.
     */
    private String importFromCSV(
        ResultSetMetaData metaData,
        String tableName,
        String[] tokens
    ) {
        StringBuilder valueBuilder = new StringBuilder(1024);
        StringBuilder columnBuilder = new StringBuilder(1024);
        try {
            int delta = tokens.length == metaData.getColumnCount() ? 0 : 1;
            for (int i = 0; i < tokens.length; i ++) {
                String columnName = metaData.getColumnName(i + 1 + delta);
                String typeName = metaData.getColumnTypeName(i + 1 + delta);
                columnBuilder.append(columnName);

                if (typeName.equalsIgnoreCase("VARCHAR") || typeName.equalsIgnoreCase("CHAR")) {
                    valueBuilder.append(
                        CommonTools.escapeString(
                            tokens[i],
                            CommonTools.SINGLE_QUOTE
                        ));
                } else {
                    valueBuilder.append(tokens[i]);
                }

                if (i != tokens.length - 1) {
                    valueBuilder.append(',');
                    columnBuilder.append(',');
                }
            }
        } catch (SQLException ex) {
            // type info is missing
            return "Missing SQL types when inserting";
        }

        ST st = getTemplate().getInstanceOf("InsertTableFromCSV");
        st.add("tableName", tableName);
        st.add("columns", columnBuilder.toString());
        st.add("values", valueBuilder.toString());
        return st.render();
    }
}
