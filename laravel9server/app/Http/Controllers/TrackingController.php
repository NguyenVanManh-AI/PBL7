<?php

namespace App\Http\Controllers;

use App\Models\Tracking;
use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Support\Facades\DB;
use App\Traits\APIResponse;
use Throwable;

class TrackingController extends Controller
{
    use APIResponse;

    public function addKeyword(Request $request)
    {
        DB::beginTransaction();
        try {
            $user = User::find(auth('user_api')->user()->id);
    
            $tracking = Tracking::firstOrCreate(
                ['user_id' => $user->id],
                ['keywords' => json_encode([])]
            );
    
            $existingKeywords = json_decode($tracking->keywords, true) ?? [];
    
            foreach ($request->keywords as $newKeyword) {
                $found = false;
                foreach ($existingKeywords as &$existingKeyword) {
                    if ($existingKeyword['name'] === $newKeyword) {
                        $existingKeyword['number'] += 1;
                        $found = true;
                        break;
                    }
                }
                if (!$found) {
                    $existingKeywords[] = [
                        'name' => $newKeyword,
                        'number' => 1
                    ];
                }
            }
    
            $tracking->keywords = json_encode($existingKeywords);
            $tracking->save();
            $tracking->keywords = json_decode($tracking->keywords);
    
            DB::commit();
            return $this->responseSuccessWithData($tracking, 'Add new keyword successfully!');
        } catch (Throwable $e) {
            DB::rollback();
            return $this->responseError($e->getMessage());
        }
    }
    public function getKeyword(Request $request)
    {
        try {
            $user = User::find(auth('user_api')->user()->id);
            $tracking = Tracking::where('user_id', $user->id)->first();
            $keywords = json_decode($tracking->keywords, true) ?? [];
            $randomKeyword = (object)[];
            if (!empty($keywords)) {
                $maxNumber = max(array_column($keywords, 'number'));
                $maxKeywords = array_filter($keywords, function($keyword) use ($maxNumber) {
                    return $keyword['number'] == $maxNumber;
                });
                $randomKeyword = $maxKeywords[array_rand($maxKeywords)];
            }
            return $this->responseSuccessWithData($randomKeyword, 'Get keyword successfully!');
        } catch (Throwable $e) {
            DB::rollback();
            return $this->responseError($e->getMessage());
        }
    }
}
