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

import com.google.common.base.Strings;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import org.jetbrains.annotations.NotNull;

import java.util.List;

public class CleanPlanJsonBuilder {
    private String url;
    private String type;
    private String sqltype;
    private String username;
    private String password;

    private boolean isCSV = false;
    private String name;
    private JsonArray table;
    private JsonArray target;
    private JsonArray value;
    private JsonArray files;

    public CleanPlanJsonBuilder csv(@NotNull String file) {
        this.isCSV = true;
        this.files = new JsonArray();
        this.files.add(new JsonPrimitive(file));
        return this;
    }

    public CleanPlanJsonBuilder url(@NotNull String url) {
        this.url = url;
        return this;
    }

    public CleanPlanJsonBuilder sqltype(@NotNull String sqltype) {
        this.sqltype = sqltype;
        return this;
    }

    public CleanPlanJsonBuilder type(@NotNull String type) {
        this.type = type;
        return this;
    }

    public CleanPlanJsonBuilder username(@NotNull String username) {
        this.username = username;
        return this;
    }

    public CleanPlanJsonBuilder password(@NotNull String password) {
        this.password = password;
        return this;
    }

    public CleanPlanJsonBuilder name(@NotNull String name) {
        this.name = name;
        return this;
    }

    public CleanPlanJsonBuilder table(@NotNull String table1) {
        this.table = new JsonArray();
        this.table.add(new JsonPrimitive(table1));
        return this;
    }

    public CleanPlanJsonBuilder target(@NotNull String table1) {
        this.target = new JsonArray();
        this.target.add(new JsonPrimitive(table1));
        return this;
    }

    public CleanPlanJsonBuilder value(String value) {
        this.value = new JsonArray();
        this.value.add(new JsonPrimitive(value));
        return this;
    }

    public CleanPlanJsonBuilder value(List<String> value) {
        this.value = new JsonArray();
        for (String v : value)
            this.value.add(new JsonPrimitive(v));
        return this;
    }

    public JsonObject build() {
        // TODO: to be improve and complete
        JsonObject source = new JsonObject();
        JsonArray rules = new JsonArray();
        JsonObject result = new JsonObject();

        if (isCSV) {
            source.add("type", new JsonPrimitive("csv"));
            source.add("file", files);
        } else {
            source.add("type", new JsonPrimitive(sqltype));
            source.add("url", new JsonPrimitive(url));
            source.add("username", new JsonPrimitive(username));
            source.add("password", new JsonPrimitive(password));
        }

        JsonObject rule = new JsonObject();
        if (!Strings.isNullOrEmpty(name))
            rule.add("name", new JsonPrimitive(name));

        rule.add("table", table);
        if (target != null)
            rule.add("target", target);

        rule.add("type", new JsonPrimitive(type));
        rule.add("value", value);
        rules.add(rule);

        result.add("source", source);
        result.add("rule", rules);

        return result;
    }
}
