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

package qa.qcri.nadeef.web.rest;

import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;
import qa.qcri.nadeef.tools.DBConfig;
import qa.qcri.nadeef.tools.Logger;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.sql.SQLDialectBase;
import qa.qcri.nadeef.web.sql.SQLUtil;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.List;

import static spark.Spark.get;

public class WidgetAction {
    private static String getSubquery(String filter) {
        String sql = "VIOLATION";
        if (!Strings.isNullOrEmpty(filter)) {
            if (filter.startsWith(":=")) {
                String vid = filter.substring(2).trim();
                if (!Strings.isNullOrEmpty(vid)) {
                    String[] tokens = vid.split(",");
                    for (String token : tokens)
                        if (!SQLUtil.isValidInteger(token))
                            throw new IllegalArgumentException("Input is not valid.");
                    sql = "(select * from VIOLATION where vid = any(array[" + vid + "])) a";
                }
            } else if (filter.startsWith("?=")) {
                String tid = filter.substring(2).trim();
                if (!Strings.isNullOrEmpty(tid)) {
                    String[] tokens = tid.split(",");
                    for (String token : tokens)
                        if (!SQLUtil.isValidInteger(token))
                            throw new IllegalArgumentException("Input is not valid.");
                    sql = "(select * from VIOLATION where tid = any(array[" + tid + "])) a";
                }
            } else {
                sql = "(select * from VIOLATION where value like '%" + filter + "%') a";
            }
        }
        return sql;
    }

    public static void setup(SQLDialect dialect) {
        SQLDialectBase dialectInstance = SQLDialectBase.createDialectBaseInstance(dialect);
        Logger tracer = Logger.getLogger(WidgetAction.class);
        get("/:project/widget/attribute", (request, response) -> {
            response.type("application/json");
            String filter = request.queryParams("filter");
            String project = request.params("project");

            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            String sql =
                String.format(
                    "select attribute, count(*) from %s group by attribute",
                    getSubquery(filter));
            return SQLUtil.query(project, sql, true);
        });

        get("/:project/widget/rule", (request, response) -> {
            response.type("application/json");
            String project = request.params("project");
            String filter = request.queryParams("filter");

            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid");

            String sql =
                String.format(
                    "select rid, count(distinct(tupleid)) as tuple, count(distinct(tablename)) as tablecount " +
                    "from %s group by rid", getSubquery(filter));

            return SQLUtil.query(project, sql, true);
        });

        get("/:project/widget/top/:k", (request, response) -> {
            response.type("application/json");
            String project = request.params("project");
            String filter = request.queryParams("filter");
            String k = request.params("k");

            if (Strings.isNullOrEmpty(project) || Strings.isNullOrEmpty(k))
                throw new IllegalArgumentException("Input is not valid");
            String sql =
                String.format(
                    "select tupleid, count(distinct(vid)) as count from %s group by tupleid " +
                    "order by count desc LIMIT %d", getSubquery(filter), Integer.parseInt(k));
            return SQLUtil.query(project, sql, true);
        });

        get("/:project/widget/violation_relation", (request, response) -> {
            response.type("application/json");
            String project = request.params("project");
            String filter = request.queryParams("filter");
            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            JsonObject countJson =
                SQLUtil.query(project, dialectInstance.countTable("violation"), true);
            JsonArray dataArray = countJson.getAsJsonArray("data");
            int count = dataArray.get(0).getAsInt();
            JsonObject result = null;
            if (count < 120000) {
                String sql =
                    "SELECT DISTINCT(VID), TUPLEID, TABLENAME FROM " + getSubquery(filter) + " ORDER BY VID";
                result = SQLUtil.query(project, sql, true);
            } else {
                result = new JsonObject();
                result.add("code", new JsonPrimitive(2));
            }
            return result;
        }, new RenderResponseTransformer());

        get("/:project/widget/overview", (request, response) -> {
            response.type("application/json");
            String project = request.params("project");
            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            JsonObject json = new JsonObject();
            JsonArray result = new JsonArray();

            DBConfig dbConfig = new DBConfig(NadeefConfiguration.getDbConfig());
            dbConfig.switchDatabase(project);

            try (
                Connection conn = DBConnectionPool.createConnection(dbConfig, true);
                Statement stat = conn.createStatement();
            ) {
                ResultSet rs = stat.executeQuery(dialectInstance.queryDistinctTable());
                List<String> tableNames = Lists.newArrayList();
                while (rs.next())
                    tableNames.add(rs.getString(1));
                rs.close();

                int sum = 0;
                for (String tableName : tableNames) {
                    rs = stat.executeQuery(dialectInstance.countTable(tableName));
                    if (rs.next())
                        sum += rs.getInt(1);
                    rs.close();
                }

                rs = stat.executeQuery(dialectInstance.countViolation());
                int vcount = 0;
                if (rs.next())
                    vcount = rs.getInt(1);
                result.add(new JsonPrimitive(sum - vcount));
                result.add(new JsonPrimitive(vcount));
                json.add("data", result);
                return json;
            } catch (Exception ex) {
                tracer.error("Query failed", ex);
                throw new RuntimeException(ex);
            }
        });
    }
}
