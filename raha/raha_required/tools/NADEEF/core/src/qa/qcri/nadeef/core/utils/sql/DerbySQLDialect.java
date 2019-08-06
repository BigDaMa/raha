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

import org.stringtemplate.v4.ST;
import org.stringtemplate.v4.STGroupFile;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.Schema;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.sql.*;

/**
 * Database manager for Apache Derby database.
 */
public class DerbySQLDialect extends SQLDialectBase {
    private static STGroupFile template =
        new STGroupFile(
            "qa*qcri*nadeef*core*utils*sql*template*DerbyTemplate.stg".replace(
            "*", "/"
            ), '$', '$');

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean supportBulkLoad() {
        return true;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int bulkLoad(DBConfig dbConfig, String tableName, Path file, boolean skipHeader) {
        Logger tracer = Logger.getLogger(DerbySQLDialect.class);
        Path inputFile = file;
        int lines = 0;
        if (skipHeader) {
            // if the header exists we need to remove it before doing loading
            try {
                // copy to a temp file.
                Path outputFile = Files.createTempFile(file.toFile().getName(), null);
                Files.copy(file, outputFile, StandardCopyOption.REPLACE_EXISTING);

                lines = removeFirstLine(outputFile);
                inputFile = outputFile;
            } catch (IOException ex)  {
                tracer.error("Creating temporary file failed.", ex);
                return 0;
            }
        }

        Connection conn = null;
        Statement stat = null;

        try {
            Schema schema = DBMetaDataTool.getSchema(dbConfig, tableName);
            StringBuilder builder = new StringBuilder();
            boolean isFirst = true;
            for (Column column : schema.getColumns()) {
                if (column.getColumnName().equalsIgnoreCase("TID"))
                    continue;
                if (!isFirst)
                    builder.append(",");
                builder.append(column.getColumnName());
                isFirst = false;
            }

            ST st = getTemplate().getInstanceOf("BulkLoad");
            st.add("schema", dbConfig.getUserName().toUpperCase());
            st.add("table", tableName.toUpperCase());
            st.add("column", builder.toString());
            st.add("filename", inputFile.toFile().getAbsolutePath());
            st.add("delimiter", ',');
            String sql = st.render();
            tracer.fine("Calling bulk loading : " + sql);

            conn = DBConnectionPool.createConnection(dbConfig, true);
            stat = conn.createStatement();
            stat.execute(sql);
        } catch (Exception ex) {
            tracer.error("Dumping file failed", ex);
            return 0;
        } finally {
            try {
                if (stat != null) {
                    stat.close();
                }

                if (conn != null) {
                    conn.close();
                }

                // remove the temporary file
                if (skipHeader) {
                    Files.delete(inputFile);
                }
            } catch (Exception ex) {}

        }
        return lines;
    }

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
            // reconstruct the SQL statement to create a table with auto-increment tid
            ResultSet rs = stat.executeQuery("SELECT * FROM " + sourceName);
            ResultSetMetaData metaData = rs.getMetaData();
            boolean hasTid = false;
            int countNum = metaData.getColumnCount();
            StringBuilder sqlBuilder = new StringBuilder(1024);
            StringBuilder columns = new StringBuilder(1024);

            for (int i = 1; i <= countNum; i ++) {
                String columnName = metaData.getColumnName(i);
                if (i != 1) {
                    sqlBuilder.append(",");
                    columns.append(",");
                }
                sqlBuilder.append(columnName)
                          .append(" ")
                          .append(metaData.getColumnTypeName(i))
                          .append("(")
                          .append(metaData.getColumnDisplaySize(i))
                          .append(") ");
                columns.append(columnName);
                if (columnName.equalsIgnoreCase("tid")) {
                    hasTid = true;
                }
            }

            if (hasTid) {
                stat.execute(
                    "CREATE TABLE " +
                    targetName +
                    " AS SELECT * FROM " +
                    sourceName +
                    " WITH NO DATA"
                );
            } else {
                ST st = getTemplate().getInstanceOf("CreateTableFromCSV");
                st.add("tableName", targetName);
                st.add("content", sqlBuilder.toString());
                String sql = st.render();
                stat.execute(sql);
            }
            String sql =
                "INSERT INTO " +
                targetName +
                " (" +
                columns.toString() +
                ") SELECT * FROM " +
                sourceName;
            stat.execute(sql);
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
        st.add("tableName", tableName.toUpperCase());
        return st.render();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String limitRow(int row) {
        return " FETCH FIRST " + row + " ROW ONLY";
    }

    private static int removeFirstLine(Path file) throws IOException {
        RandomAccessFile raf = new RandomAccessFile(file.toFile(), "rw");
        //Initial write position
        long writePosition = raf.getFilePointer();
        raf.readLine();
        // Shift the next lines upwards.
        long readPosition = raf.getFilePointer();

        byte[] buff = new byte[40960];
        int n;
        int size = 0;
        while (-1 != (n = raf.read(buff))) {
            size += n;
            raf.seek(writePosition);
            raf.write(buff, 0, n);
            readPosition += n;
            writePosition += n;
            raf.seek(readPosition);
        }

        // TODO: deal with non-eol?
        // String eol = System.lineSeparator();
        // raf.writeChars(eol);
        raf.setLength(writePosition);
        raf.close();
        return size;
    }
}
