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
    "state",
    "render",
    "table",
    "requester",
    "text!mvc/template/home.template.html",
    "text!mvc/template/table.template.html",
    "text!mvc/template/tab.template.html",
    "mvc/ControllerView"
], function (
    State,
    Renderer,
    Table,
    Requester,
    HomeTemplate,
    TableTemplate,
    WidgetTemplate,
    ControllerView
) {
    "use strict";
    function info(msg) {
        $('home-alert-info').alert('close');
        $('#home-alert').html([
            ['<div class="alert alert-success" id="home-alert-info">'],
            ['<button type="button" class="close" data-dismiss="alert">'],
            ['&times;</button>'],
            ["<span><h4>" + msg + "</h4></span></div>"]
        ].join(''));
    
        window.setTimeout(function () { $('#home-alert-info').alert('close'); }, 2000);
    }
    
    function err(msg) {
        $('#home-alert').html([
            ['<div class="alert alert-error">'],
            ['<button type="button" class="close" data-dismiss="alert">'],
            ['&times;</button>'],
            ['<span>' + msg + '</span></div>']
        ].join(''));
    }

    function bindEvent() {
        $('#widget').find("ul li").on('click', function (e) {
            renderWidget(e.currentTarget.id);
        });

        $('#table-tabs a[data-toggle="tab"]').on('click', function (e) {
            var tableName = e.target.text.toLowerCase();
            renderTable(tableName);
        });

        $('#clear').on('click', function () {
            Requester.deleteViolation({
                success: function () {
                    info('Violations are removed');
                    refresh();
                },
                failure: function () {
                    err("Removing violations failed.");
                }
            });
        });
    }

    function refresh() {
        var defaultWidget = $('#widget').find('li.active');
        if (defaultWidget != null) {
            renderWidget(defaultWidget[0].id);
        }

        var activeTable = $("#table-tabs").find('.active a')[0].text;
        if (activeTable != null) {
            if (activeTable.indexOf('Violation') > -1) {
                renderTable('violation', true);
            } else if (activeTable.indexOf('Audit') > -1) {
                renderTable('audit', true);
            } else if (activeTable.indexOf('Source') > -1) {
                renderTable('source', true);
            }
        }
    }
    
    function start() {
        render();
        bindEvent();
        // render default view
        var defaultWidget = $('#widget').find('li.active');
        if (defaultWidget != null && defaultWidget.length > 0) {
            renderWidget(defaultWidget[0].id);
        } else {
            renderWidget("overview");
        }
    }

    function renderTable(id, reload) {
        console.log('Load table ' + id);
        switch (id) {
        case 'violation':
            var rule = $("#selected_rule").val();
            if (!_.isEmpty(rule)) {
                Table.load({domId: 'violation-table', table: 'violation', rule: rule}, reload);
            }
            break;
        case 'audit':
            Table.load({domId : 'audit-table', table : 'audit'}, reload);
            break;
        case 'source':
            var sourceTable = $("#selected_source").val();
            if (!_.isEmpty(sourceTable)) {
                Table.load({domId: 'source-table', table: sourceTable}, reload);
            }
            break;
        }
    }
    
    function renderWidget(id) {
        console.log('Render ' + id);
        switch (id) {
        case 'tab_overview':
            Renderer.drawOverview('overview');
            break;
        case 'tab_attribute':
            Renderer.drawAttribute('attribute');
            break;
        case 'tab_distribution':
            Renderer.drawDistribution('distribution');
            break;
        case 'tab_tupleRank':
            Renderer.drawTupleRank('tupleRank');
            break;
        case 'tab_violationRelation':
            Renderer.drawViolationRelation('violationRelation');
            break;
        }
    }
    
    function render() {
        var widgetTabs = {
            tabs: [
                {tag : "overview", head : "Overview", isActive : true},
                {tag : "attribute", head : "Rule Attribute", isActive : false},
                {tag : "distribution", head : "Rule Distribution", isActive : false},
                {tag : "violationRelation", head : "Sky Graph", isActive : false},
                {tag : "tupleRank", head : "Tuple Rank", isActive : false}
            ]
        };

        var homeHtml = _.template(
            HomeTemplate, {
                placeholder1 : _.template(WidgetTemplate)(widgetTabs),
                placeholder2 : _.template(TableTemplate)()
            });

        $("#container").html(homeHtml);
        
        // render the cleanplan controller
        ControllerView.render('controller');
    }
    
    return {
        start: start,
        refresh: refresh
    };
});
