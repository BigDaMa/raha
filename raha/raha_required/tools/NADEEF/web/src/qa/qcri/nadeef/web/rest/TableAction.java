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
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import qa.qcri.nadeef.tools.sql.SQLDialect;
import qa.qcri.nadeef.web.sql.SQLDialectBase;
import qa.qcri.nadeef.web.sql.SQLUtil;

import java.util.ArrayList;

import static spark.Spark.delete;
import static spark.Spark.get;

public class TableAction {
    public static void setup(SQLDialect dialect) {
        SQLDialectBase dialectInstance = SQLDialectBase.createDialectBaseInstance(dialect);
        get("/:project/violation/metadata", (x, response) -> {
            String project = x.params("project");
            String rule = x.queryParams("rule");

            if (Strings.isNullOrEmpty(project) || Strings.isNullOrEmpty(rule))
                throw new IllegalArgumentException("Input is not valid");
            String sql = String.format(
                "select count(*), tablename from violation " +
                "where rid = '%s' group by tablename", rule
            );
            return SQLUtil.query(project, sql, true);
        });

        get("/:project/violation/:tablename", (x, response) -> {
            String project = x.params("project");
            String rule = x.queryParams("rule");
            String tableName = x.params("tablename");

            if (Strings.isNullOrEmpty(project) ||
                Strings.isNullOrEmpty(rule) ||
                Strings.isNullOrEmpty(tableName))
                throw new IllegalArgumentException("Input is not valid");

            String start_ = x.queryParams("start");
            String interval_ = x.queryParams("length");
            String filter = x.queryParams("search[value]");

            if (!(
                SQLUtil.isValidInteger(start_) &&
                SQLUtil.isValidInteger(interval_)
            )) throw new IllegalArgumentException("Input is not valid.");

            String vidFilter = "";
            String tidFilter = "";
            String columnFilter = "";
            if (!Strings.isNullOrEmpty(filter)) {
                if (filter.startsWith(":=")) {
                    vidFilter = filter.substring(2).trim();
                    if (!Strings.isNullOrEmpty(vidFilter)) {
                        String[] tokens = vidFilter.split(",");
                        for (String token : tokens)
                            if (!SQLUtil.isValidInteger(token))
                                throw new IllegalArgumentException("Input is not valid.");
                        vidFilter = "and vid = any(array[" + vidFilter + "])";
                    }
                } else  if (filter.startsWith("?=")) {
                    tidFilter = filter.substring(2).trim();
                    if (!Strings.isNullOrEmpty(tidFilter)) {
                        String[] tokens = tidFilter.split(",");
                        for (String token : tokens)
                            if (!SQLUtil.isValidInteger(token))
                                throw new IllegalArgumentException("Input is not valid.");
                        tidFilter = "and tupleid = any(array[" + tidFilter + "])";
                    }
                } else {
                    columnFilter = "and value like '%" + filter + "%' ";
                }
            }

            int start = Strings.isNullOrEmpty(start_) ? 0 : Integer.parseInt(start_);
            int interval =
                Strings.isNullOrEmpty(interval_) ? 10 : Integer.parseInt(interval_);

            String rawSql = String.format(
                "select a.*, b.vid, b._attrs from %s a inner join " +
                    "(select vid, tupleid, array_agg(attribute) as _attrs from violation " +
                    "where rid='%s' and tablename = '%s' %s %s %s group by vid, tupleid) b " +
                    "on a.tid = b.tupleid order by vid",
                tableName,
                rule,
                tableName,
                vidFilter,
                tidFilter,
                columnFilter);

            String limitSql = String.format("%s limit %d offset %d", rawSql, interval, start);
            JsonObject result = SQLUtil.query(project, limitSql, true);
            String countSql = String.format("select count(*) from (%s) a", rawSql);
            JsonObject countJson = SQLUtil.query(project, countSql, false);
            JsonArray dataArray = countJson.getAsJsonArray("data");
            int count = dataArray.get(0).getAsInt();
            result.add("iTotalRecords", new JsonPrimitive(count));
            result.add("iTotalDisplayRecords", new JsonPrimitive(count));
            if (x.queryParams("sEcho") != null)
                result.add("sEcho", new JsonPrimitive(x.queryParams("sEcho")));
            return result;
        });

        /**
         * Gets data table with pagination support.
         */
        get("/:project/table/:tablename", (x, response) -> {
            String tableName = x.params("tablename");
            String project = x.params("project");

            if (Strings.isNullOrEmpty(tableName) || Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid");

            JsonObject queryJson;
            String start_ = x.queryParams("start");
            String interval_ = x.queryParams("length");

            if (!(
                SQLUtil.isValidInteger(start_) &&
                SQLUtil.isValidInteger(interval_)
            )) throw new IllegalArgumentException("Input is not valid.");

            int start = Strings.isNullOrEmpty(start_) ? 0 : Integer.parseInt(start_);
            int interval =
                Strings.isNullOrEmpty(interval_) ? 10 : Integer.parseInt(interval_);
            String filter = x.queryParams("search[value]");
            ArrayList columns = null;
            if (!Strings.isNullOrEmpty(filter)) {
                JsonObject objSchema =
                    SQLUtil.query(project, dialectInstance.querySchema(tableName), true);
                columns =
                    new Gson().fromJson(
                        objSchema.getAsJsonArray("schema"),
                        ArrayList.class
                    );
            }

            queryJson =
                SQLUtil.query(
                    project,
                    dialectInstance.queryTable(
                        tableName,
                        start,
                        interval,
                        columns,
                        filter
                    ),
                    true
                );

            JsonObject countJson =
                SQLUtil.query(project, dialectInstance.countTable(tableName), true);
            JsonArray dataArray = countJson.getAsJsonArray("data");
            int count = dataArray.get(0).getAsInt();
            queryJson.add("iTotalRecords", new JsonPrimitive(count));
            queryJson.add("iTotalDisplayRecords", new JsonPrimitive(count));
            if (x.queryParams("sEcho") != null)
                queryJson.add("sEcho", new JsonPrimitive(x.queryParams("sEcho")));
            return queryJson;
        });

        get("/:project/table/:tablename/schema", (request, response) -> {
            String tableName = request.params("tablename");
            String project = request.params("project");

            if (Strings.isNullOrEmpty(tableName) || Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            return SQLUtil.query(project, dialectInstance.querySchema(tableName), true);
        });

        delete("/:project/table/violation", (request, response) -> {
            String project = request.params("project");
            if (Strings.isNullOrEmpty(project))
                throw new IllegalArgumentException("Input is not valid.");

            return SQLUtil.update(project, dialectInstance.deleteViolation());
        });
    }
}
