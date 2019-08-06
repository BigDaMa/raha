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

define([
    "render",
    "text!mvc/template/tab.template.html",
    "text!mvc/template/dashboard.template.html"
], function (renderer, widgetTemplate, dashboardTemplate) {
    "use strict";
    function start() {
        render();

        renderer.drawOverview('overview');
        renderer.drawAttribute('attribute');
        renderer.drawDistribution('distribution');
		renderer.drawTupleRank('tupleRank');
    }
    
    function render() {
        var tabTemplate = _.template(widgetTemplate);
        var overviewHtml =
            tabTemplate({tabs:
                [{tag : "overview", head : "Overview", isActive : true}]
            });
        var attributeHtml =
            tabTemplate({tabs:
                [{tag : "attribute", head : "Rule Attribute", isActive : true}]
            });
        var distributionHtml =
            tabTemplate({tabs:
                [{tag : "distribution",
                  head : "Rule Distribution",
                  isActive : true}]
            });
        var tupleRankHtml =
            tabTemplate({tabs:
                [{tag : "tupleRank",
                  head : "Tuple Rank",
                  isActive : true}]
            });
        
        var dashboardTemplate = _.template(dashboardTemplate);
        var dashboardHtml = dashboardTemplate(
            {
                placeholder1: overviewHtml,
                placeholder2: attributeHtml,
                placeholder4: distributionHtml,
                placeholder3: tupleRankHtml
            });

        $('#container').html(dashboardHtml);
    }
    
    return {
        start: start
    };
});
