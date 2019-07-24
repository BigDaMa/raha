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
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

class RuleJsonAdapter {
    public String type;
    public String name;
    public List<String> table;
    public List<String> target;
    public List<String> value;

    public boolean hasName() {
        return !Strings.isNullOrEmpty(name);
    }

    public boolean hasTable() {
        return table != null && table.size() > 0;
    }

    public boolean hasTarget() {
        return target != null && target.size() > 0;
    }

}

class CleanPlanJsonAdapter {
    public CleanPlanJsonAdapter(DBConfigJsonAdapter dbConfig, List<RuleJsonAdapter> rules) {
        this.dbConfig = dbConfig;
        this.rules = rules;
    }

    public DBConfigJsonAdapter dbConfig;
    public List<RuleJsonAdapter> rules;
}

class DBConfigJsonAdapter {
    public String url;
    public String type;
    public String username;
    public String password;
    public List<String> file;

    public boolean isCSV() {
        return type.equalsIgnoreCase("csv");
    }
}

class CleanPlanJsonDeserializer implements JsonDeserializer<CleanPlanJsonAdapter> {
    @Override
    public CleanPlanJsonAdapter deserialize(
        JsonElement jsonElement,
        Type type,
        JsonDeserializationContext jsonDeserializationContext
    ) throws JsonParseException {
        Gson gson = new Gson();
        JsonObject rootObj = (JsonObject)jsonElement;
        DBConfigJsonAdapter dbConfigJson =
            gson.fromJson(rootObj.get("source"), DBConfigJsonAdapter.class);

        if (Strings.isNullOrEmpty(dbConfigJson.type))
            throw new JsonParseException("Data source type is missing.");

        if (dbConfigJson.type.equalsIgnoreCase("csv") &&
            (dbConfigJson.file == null || dbConfigJson.file.size() == 0))
            throw new JsonParseException("File path is missing.");

        if ((!dbConfigJson.isCSV()) &&
            (Strings.isNullOrEmpty(dbConfigJson.url) ||
            Strings.isNullOrEmpty(dbConfigJson.username)))
            throw new JsonParseException("Database source is not correct.");

        List<RuleJsonAdapter> ruleJson =
            gson.fromJson(rootObj.get("rule"), new TypeToken<List<RuleJsonAdapter>>(){}.getType());

        if (ruleJson == null || ruleJson.size() == 0)
            throw new JsonParseException("Rule cannot be empty.");
        for (RuleJsonAdapter rule : ruleJson) {
            if (Strings.isNullOrEmpty(rule.type) ||
                (rule.hasTable() && rule.table.size() > 2) ||
                rule.value == null || rule.value.size() == 0)
                throw new JsonParseException("Rule is not correct.");

            if (dbConfigJson.isCSV()) {
                // check the table name correctness
                List<String> fullFileNames = Lists.newArrayList();
                for (String fullFileName : dbConfigJson.file)
                    fullFileNames.add(Files.getNameWithoutExtension(fullFileName));
                if (rule.hasTable()) {
                    for (String file : rule.table)
                        if (!fullFileNames.contains(file))
                            throw new JsonParseException("Invalid table name " + file + ".");
                }
            }

            if (rule.hasTarget() && rule.table.size() != rule.target.size())
                throw new JsonParseException("Table and Target tables do not have the same size.");
            if (rule.hasTable() && rule.table.size() > 2)
                throw new JsonParseException("Rule cannot work with more than 2 tables.");
        }

        return new CleanPlanJsonAdapter(dbConfigJson, ruleJson);
    }
}