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

import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import spark.ResponseTransformer;

public class RenderResponseTransformer implements ResponseTransformer {
    @Override
    public String render(Object o) throws Exception {
        JsonObject result;
        if (o == null || !(o instanceof JsonObject))
            result = new JsonObject();
        else
            result = (JsonObject)o;

        if (!result.has("code"))
            result.add("code", new JsonPrimitive(1));
        return result.toString();
    }
}
