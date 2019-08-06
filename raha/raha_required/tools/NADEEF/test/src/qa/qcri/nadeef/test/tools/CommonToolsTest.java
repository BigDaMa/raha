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

package qa.qcri.nadeef.test.tools;

import org.junit.Assert;
import org.junit.Test;
import qa.qcri.nadeef.tools.CommonTools;

public class CommonToolsTest {
    @Test public void escapeTest() {
        char enclosed = '"';
        String in, out;
        in = "abcde";
        out = CommonTools.escapeString(in, enclosed);
        Assert.assertEquals("\"abcde\"", out);
        Assert.assertEquals(in, CommonTools.unescapeString(out, enclosed));

        in = "\"";
        out = CommonTools.escapeString(in, enclosed);
        Assert.assertEquals("\"\\\"\"", out);
        Assert.assertEquals(in, CommonTools.unescapeString(out, enclosed));

        in = "";
        out = CommonTools.escapeString(in, enclosed);
        Assert.assertEquals("\"\"", out);
        Assert.assertEquals("", CommonTools.unescapeString(out, enclosed));

        in = "Haven't said \"yes\"";
        out = CommonTools.escapeString(in, enclosed);
        Assert.assertEquals("\"Haven't said \\\"yes\\\"\"", out);
        Assert.assertEquals(in, CommonTools.unescapeString(out, enclosed));
    }
}
