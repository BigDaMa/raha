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
    'state',
    'requester',
    'text!mvc/template/progressbar.template.html'],
    function(State, Requester, ProgressBarTemplate) {
        var intervalId = null;

        function info(msg) {
            $('#home-alert-info').alert('close');
            $('#home-alert').html([
                ['<div class="alert alert-success" id="home-alert-info">'],
                ['<button type="button" class="close" data-dismiss="alert">'],
                ['&times;</button>'],
                ['<span><h3>' + msg + '</h3></span></div>']].join(''));

            window.setTimeout(function() { $('#home-alert-info').alert('close'); }, 2000);
        }

        function updateProgress(id) {
            var jobList = State.get('job');
            // TODO: fix progressbar rendering twice problem
            if (jobList.length == 0) {
                if ($('#' + id).find("#progressbar-content").length == 0) {
                    var html =
                        _.template(ProgressBarTemplate)({ progress: [] });
                    $('#' + id).html(html);
                }
                return;
            }

            Requester.getProgress({
                blockUI: false,
                success: function(data) {
                    var values = [];
                    if (jobList && jobList.length > data['data'].length)
                        info("There are Job(s) finished, please do refresh to see the result.");
                    var runningList = data['data'];
                    var runningKeySet = {};
                    _.each(runningList, function(job) {
                        runningKeySet[job['key']] = job;
                    });

                    var newJobList = [];
                    for (var i = 0; i < jobList.length; i ++) {
                        var job = jobList[i];
                        if (job.key in runningKeySet) {
                            var v = runningKeySet[job.key];
                            values.push({
                                name : job.name,
                                value : v['overallProgress']
                            });
                            newJobList.push(job);
                        }
                    }

                    State.set('job', newJobList);
                    var html =
                        _.template(ProgressBarTemplate)({ progress: values });
                    $('#' + id).html(html);
                },
                failure: function(response) {
                    console.log("Requesting progress failed : " + response.responseText);
                }}
            );
        }

		function start(id) {
            if (intervalId != null)
                clearInterval(intervalId);
            updateProgress(id);
            intervalId = setInterval(function() { updateProgress(id); }, 3000);
		}

		return {
			start: start
		};
	}
);
