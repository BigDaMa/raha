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
import com.google.common.collect.Lists;
import qa.qcri.nadeef.core.datamodel.Cell;
import qa.qcri.nadeef.core.datamodel.Column;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.core.utils.sql.SQLDialectBase;
import qa.qcri.nadeef.core.utils.sql.SQLDialectFactory;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.io.*;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Collection;
import java.util.List;

/**
 * Violation Extension helper.
 */
public class Violations {
    private static Logger tracer = Logger.getLogger(Violations.class);

    /**
     * Generates violation row number from the database.
     * @return new unique violation id.
     */
    public static int getViolationRowCount(DBConfig nadeefConfig) throws Exception {
        Connection conn = null;
        int result = 0;
        try {
            conn = DBConnectionPool.createConnection(nadeefConfig, true);
            result = getViolationRowCount(conn);
        } finally {
            if (conn != null) {
                conn.close();
            }
        }
        return result;
    }

    /**
     * Generates violation id from the database.
     * @return new unique violation id.
     */
    public static int generateViolationId(DBConnectionPool pool) throws Exception {
        Connection conn = pool.getNadeefConnection();
        conn.setAutoCommit(true);
        int result = -1;
        try {
            result = generateViolationId(conn);
        } finally {
            conn.close();
        }
        return result;
    }

    /**
     * Generates violation id from the database.
     * @return new unique violation id.
     */
    public static int generateViolationId(DBConfig dbConfig) throws Exception {
        Connection conn = DBConnectionPool.createConnection(dbConfig);
        int result = -1;
        try {
            result = generateViolationId(conn);
        } finally {
            if (conn != null) {
                conn.close();
            }
        }
        return result;
    }

    private static int generateViolationId(Connection conn) throws Exception {
        Statement stat = null;
        ResultSet rs = null;
        int result = -1;
        try {
            stat = conn.createStatement();
            String tableName = NadeefConfiguration.getViolationTableName();
            SQLDialectBase dialectManager =
                SQLDialectFactory.getNadeefDialectManagerInstance();
            rs = stat.executeQuery(dialectManager.nextVid(tableName));
            if (rs.next()) {
                result = rs.getInt("vid");
            }
        } finally {
            if (rs != null) {
                rs.close();
            }

            if (stat != null) {
                stat.close();
            }

            if (conn != null) {
                conn.close();
            }
        }
        return result;
    }

    private static int getViolationRowCount(Connection conn) throws Exception {
        Statement stat = null;
        ResultSet rs = null;
        int result = -1;
        try {
            SQLDialectBase dialectManager =
                SQLDialectFactory.getNadeefDialectManagerInstance();
            String tableName = NadeefConfiguration.getViolationTableName();

            stat = conn.createStatement();
            String sql = dialectManager.countTable(tableName);
            rs = stat.executeQuery(sql);
            tracer.fine(sql);
            if (rs.next()) {
                result = rs.getInt(1);
            } else {
                result = 0;
            }
        } catch (SQLException ex) {
            tracer.error("Violation counts failed.", ex);
        } finally {
            if (rs != null) {
                rs.close();
            }

            if (stat != null) {
                stat.close();
            }
        }
        return result;
    }

    /**
     * Generates a list of violations from a query result.
     * @param resultSet query result.
     * @return a list of violations.
     */
    public static Collection<Violation> fromQuery(ResultSet resultSet)
        throws SQLException {
        Preconditions.checkNotNull(resultSet);
        List<Violation> result = Lists.newArrayList();
        int lastVid = -1;
        Violation violation = null;
        while (resultSet.next()) {
            int vid = resultSet.getInt("vid");
            String rid = resultSet.getString("rid");
            String tableName = resultSet.getString("tablename");
            int tupleId = resultSet.getInt("tupleid");
            String attribute = resultSet.getString("attribute");
            String value = resultSet.getString("value");
            Column column = new Column(tableName, attribute);
            Cell cell = new Cell(column, tupleId, value);
            if (vid != lastVid || vid == -1) {
                violation = new Violation(rid, vid);
                violation.addCell(cell);
                result.add(violation);
                lastVid = vid;
            } else {
                violation.addCell(cell);
            }
        }
        return result;
    }

    public static Collection<Violation> fromCSV(File csvFile)
        throws IOException {
        Preconditions.checkNotNull(csvFile);
        BufferedReader reader = new BufferedReader(new FileReader(csvFile));
        // skip the head
        String line = reader.readLine();
        List<Violation> result = Lists.newArrayList();
        int lastVid = -1;
        Violation violation = null;
        while ((line = reader.readLine()) != null) {
            String[] token = line.split(";");
            if (token.length != 6) {
                throw
                    new InvalidObjectException("The given CSV is not a valid violation CSV file.");
            }
            int vid = Integer.parseInt(token[0]);
            String rid = token[1].replace("\"", "");
            String tableName = token[2].replace("\"", "");
            int tupleId = Integer.parseInt(token[3]);
            String attribute = token[4].replace("\"", "");
            String value = token[5].replace("\"", "");
            Column column = new Column(tableName, attribute);
            Cell cell = new Cell(column, tupleId, value);
            if (vid != lastVid || vid == -1) {
                violation = new Violation(rid);
                violation.addCell(cell);
                result.add(violation);
                lastVid = vid;
            } else {
                violation.addCell(cell);
            }
        }
        return result;
    }
}
