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

package qa.qcri.nadeef.web;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class HTTPPostJsonParser {
    public static HashMap<String, Object> parse(String body) {
        HashMap<String, Object> result = new HashMap<>();

        String[] tokens = body.split("&");
        for (String token : tokens) {
            String[] pair = token.split("=");
            int index = pair[0].indexOf("%5B%5D");
            String key = index > -1 ? pair[0].substring(0, index) : pair[0];
            if (result.containsKey(key)) {
                @SuppressWarnings("unchecked")
                List<String> container = (List<String>)result.get(key);
                container.add(pair[1]);
            } else if (index > -1) {
                List<String> list = new ArrayList<>();
                list.add(pair[1]);
                result.put(key, list);
            } else {
                result.put(key, pair[1]);
            }
        }

        return result;
    }
}
