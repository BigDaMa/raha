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

package qa.qcri.nadeef.web.sql;

import com.google.common.base.Strings;
import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;

import java.sql.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SQLUtil {
    private static Logger tracer = Logger.getLogger(SQLUtil.class);

    public static boolean isValidTableName(String s) {
        boolean isGood = true;
        if (!Strings.isNullOrEmpty(s)) {
            Pattern pattern = Pattern.compile("\\w+");
            Matcher matcher = pattern.matcher(s);
            if (!matcher.find())
                isGood = false;
        }
        return isGood;
    }

    public static boolean isValidInteger(String s) {
        boolean isGood = true;
        if (!Strings.isNullOrEmpty(s)) {
            try {
                int ignore = Integer.parseInt(s);
            } catch (Exception ex) {
                isGood = false;
            }
        }
        return isGood;
    }

    //<editor-fold desc="Private helpers">
    public static JsonObject query(String dbName, String sql, boolean includeHeader)
        throws RuntimeException {
        DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
        dbConfig.switchDatabase(dbName);

        try (Connection conn = DBConnectionPool.createConnection(dbConfig, true);
             Statement stat = conn.createStatement();
             ResultSet rs = stat.executeQuery(sql)
        ) {
            return queryToJson(rs, includeHeader);
        } catch(Exception ex) {
            tracer.error("Exception on query " + sql, ex);
            throw new RuntimeException(ex);
        }
    }

    public static JsonObject update(String dbname, String sql) throws RuntimeException {
        DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
        dbConfig.switchDatabase(dbname);
        try (Connection conn = DBConnectionPool.createConnection(dbConfig, true);
             Statement stat = conn.createStatement()){
            stat.execute(sql);
            JsonObject obj = new JsonObject();
            obj.add("data", new JsonPrimitive(0));
            return obj;
        } catch (
            SQLException | ClassNotFoundException |
                InstantiationException | IllegalAccessException ex) {
            tracer.error("Exception", ex);
            throw new RuntimeException(ex.getMessage(), ex);
        }
    }

    public static JsonObject queryToJson(ResultSet rs, boolean includeHeader)
        throws RuntimeException {
        try {
            ResultSetMetaData metaData = rs.getMetaData();
            int ncol = metaData.getColumnCount();
            JsonObject queryObject = new JsonObject();
            if (includeHeader) {
                JsonArray array = new JsonArray();
                for (int i = 1; i <= ncol; i++)
                    array.add(new JsonPrimitive(metaData.getColumnName(i)));

                queryObject.add("schema", array);
            }

            JsonArray data = new JsonArray();
            while (rs.next()) {
                JsonArray entry = new JsonArray();
                for (int i = 1; i <= ncol; i++) {
                    Object obj = rs.getObject(i);
                    if (obj != null)
                        entry.add(new JsonPrimitive(obj.toString()));
                    else
                        entry.add(JsonNull.INSTANCE);
                }
                data.add(entry);
            }

            queryObject.add("data", data);
            return queryObject;
        } catch (SQLException ex) {
            tracer.error("Exception", ex);
            throw new RuntimeException(ex);
        }
    }
}
