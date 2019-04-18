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

import com.google.common.base.Stopwatch;
import org.postgresql.copy.CopyManager;
import org.postgresql.core.BaseConnection;
import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.io.FileReader;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.concurrent.TimeUnit;

/**
 * Database manager for Apache Derby database.
 */
public class PostgresSQLDialect extends SQLDialectBase {
    public static STGroupFile template =
        new STGroupFile("qa/qcri/nadeef/core/utils/sql/template/PostgresTemplate.stg", '$', '$');

    /**
     * {@inheritDoc}
     */
    @Override
    protected STGroupFile getTemplate() {
        return template;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void copyTable(Connection conn, String sourceName, String targetName)
            throws SQLException {
        Statement stat = null;
        try {
            stat = conn.createStatement();
            stat.execute("SELECT * INTO " + targetName + " FROM " + sourceName);
            ResultSet rs = stat.executeQuery(
                "select * from information_schema.columns where table_name = '" +
                    targetName + "' and column_name = 'tid'"
            );

            if (!rs.next()) {
                stat.execute("alter table " + targetName + " add column tid serial primary key");
            }
        } finally {
            if (stat != null) {
                stat.close();
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String countTable(String tableName) {
        ST st = getTemplate().getInstanceOf("CountTable");
        st.add("tableName", tableName);
        return st.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String limitRow(int row) {
        return " LIMIT " + row;
    }

    @Override public boolean supportBulkLoad() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int bulkLoad(
        DBConfig dbConfig,
        String tableName,
        Path file,
        boolean skipHeader
    ) {
        Logger tracer = Logger.getLogger(PostgresSQLDialect.class);
        tracer.info("Bulk load CSV file " + file.toString());
        try (Connection conn = DBConnectionPool.createConnection(dbConfig, true);
             FileReader reader = new FileReader(file.toFile())
        ) {
            Stopwatch watch = Stopwatch.createStarted();
            Schema schema = DBMetaDataTool.getSchema(dbConfig, tableName);
            StringBuilder builder = new StringBuilder();
            for (Column column : schema.getColumns()) {
                if (column.getColumnName().equalsIgnoreCase("TID"))
                    continue;
                builder.append(column.getColumnName()).append(",");
            }
            builder.deleteCharAt(builder.length() - 1);

            CopyManager copyManager = new CopyManager((BaseConnection)conn);
            String sql =
                String.format(
                    "COPY %s (%s) FROM STDIN WITH (FORMAT 'csv', DELIMITER ',', HEADER %s)",
                    tableName,
                    builder.toString(),
                    skipHeader ? "true" : "false");
            tracer.info(sql);
            copyManager.copyIn(sql, reader);
            watch.stop();
            tracer.info("Bulk load finished in " + watch.elapsed(TimeUnit.MILLISECONDS) + " ms");
        } catch (Exception ex) {
            tracer.error("Loading csv file " + file.getFileName() + " failed.", ex);
            return 1;
        }
        return 0;
    }
}
